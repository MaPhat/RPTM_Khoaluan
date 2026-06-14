from __future__ import print_function
from __future__ import division
import os
import os.path as osp
import time
import torch
import numpy as np
import numpy.ma as ma
import random
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

from eval import do_test
from utils.loggers import RankLogger
from utils.torchtools import accuracy, save_checkpoint
from utils.functions import search, strint
from utils.avgmeter import AverageMeter
from utils.visualtools import visualize_ranked_results
from utils.graph_reranking import *
from tqdm import tqdm

def do_train(cfg, trainloader, train_dict, data_tfr, testloader_dict, dm,
             model, optimizer, scheduler, criterion_htri,criterion_xent, gcn_model=None):
    ranklogger = RankLogger(cfg.DATASET.SOURCE_NAME, cfg.DATASET.TARGET_NAME)
    use_gpu = cfg.MISC.USE_GPU
    gms = train_dict['gms']
    pidx = train_dict['pidx']
    folders = []
    for fld in os.listdir(cfg.DATASET.SPLIT_DIR):
        folders.append(fld)
    # data_index = search_index(gms, cfg.DATASET.SPLIT_DIR, folders)
    data_index = search(cfg.DATASET.SPLIT_DIR)

    best_rank1 = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        losses = AverageMeter()
        xent_losses = AverageMeter()
        htri_losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        if gcn_model is not None:
            gcn_model.train()
        model.train()
        for p in model.parameters():
            p.requires_grad = True  # open all layers

        end = time.time()
        for batch_idx, (img, label, index, pid, cid) in enumerate(tqdm(trainloader)):

            trainX, trainY = torch.zeros((cfg.SOLVER.TRAIN_BATCH_SIZE * 3, 3, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH), dtype=torch.float32), torch.zeros(
                (cfg.SOLVER.TRAIN_BATCH_SIZE * 3), dtype=torch.int64)

            for i in range(cfg.SOLVER.TRAIN_BATCH_SIZE):

                labelx = str(label[i])
                indexx = int(index[i])
                cidx = int(pid[i])
                if indexx > len(gms[labelx]) - 1:
                    indexx = len(gms[labelx]) - 1
                a = gms[labelx][indexx]

                if cfg.MODEL.RPTM_SELECT == 'min':
                    threshold = np.arange(10)
                elif cfg.MODEL.RPTM_SELECT == 'mean':
                    threshold = np.arange(np.amax(gms[labelx][indexx])//2)
                elif cfg.MODEL.RPTM_SELECT == 'max':
                    threshold = np.arange(np.amax(gms[labelx][indexx]))
                else:
                    threshold = np.arange(np.amax(gms[labelx][indexx]) // 2) #defaults to mean

                minpos = np.argmin(ma.masked_where(np.isin(a, threshold), a))
                pos_dic = data_tfr[data_index[cidx][1] + minpos]
                # print(pos_dic[1])
                # Pick a random negative label from the available gms keys, different from current
                gms_keys = list(gms.keys())
                while True:
                    negative_label = random.choice(gms_keys)
                    if negative_label != labelx:
                        break
                neg_cid = pidx.get(negative_label, random.choice(list(pidx.values())))
                neg_index = random.choice(range(0, len(gms[negative_label])))

                neg_dic = data_tfr[data_index[neg_cid][1] + neg_index]
                trainX[i] = img[i]
                trainX[i + cfg.SOLVER.TRAIN_BATCH_SIZE] = pos_dic[0]
                trainX[i + (cfg.SOLVER.TRAIN_BATCH_SIZE * 2)] = neg_dic[0]
                trainY[i] = cidx
                trainY[i + cfg.SOLVER.TRAIN_BATCH_SIZE] = pos_dic[3]
                trainY[i + (cfg.SOLVER.TRAIN_BATCH_SIZE * 2)] = neg_dic[3]
            optimizer.zero_grad()
            trainX = trainX.cuda()
            trainY = trainY.cuda()
            outputs, features = model(trainX)

            if gcn_model is not None:
                B = cfg.SOLVER.TRAIN_BATCH_SIZE
                
                # Build graphs and refine ALL features (anchor, positive, negative)
                cam = cid  # use real camera IDs, not image indices
                # Replicate cam for positive and negative samples
                cam_all = torch.cat([cam, cam, cam], dim=0)
                
                A_g, A_c = build_graphs_for_batch(features, cam_all)
                A_g_norm = normalize_adj(A_g)
                if A_c.sum() == 0 or torch.isnan(A_c).any():
                    A_c_norm = torch.zeros_like(A_c)
                else:
                    A_c_norm = normalize_adj(A_c)
                refined_global = gcn_model(features, A_g_norm, A_c_norm)

                gcn_recon_loss = torch.nn.functional.mse_loss(refined_global, features.detach())

                xent_loss = criterion_xent(outputs[0:cfg.SOLVER.TRAIN_BATCH_SIZE], trainY[0:cfg.SOLVER.TRAIN_BATCH_SIZE])
                htri_loss = criterion_htri(refined_global, trainY)

                gcn_weight = 0.05
                loss = cfg.LOSS.LAMBDA_HTRI * htri_loss + cfg.LOSS.LAMBDA_XENT * xent_loss + gcn_weight * gcn_recon_loss if gcn_model is not None else cfg.LOSS.LAMBDA_HTRI * htri_loss + cfg.LOSS.LAMBDA_XENT * xent_loss
            else:
                xent_loss = criterion_xent(outputs[0:cfg.SOLVER.TRAIN_BATCH_SIZE], trainY[0:cfg.SOLVER.TRAIN_BATCH_SIZE])
                htri_loss = criterion_htri(features, trainY) 
                loss = cfg.LOSS.LAMBDA_HTRI * htri_loss + cfg.LOSS.LAMBDA_XENT * xent_loss

            if cfg.SOLVER.USE_AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            for param_group in optimizer.param_groups:
                # print(param_group['lr'] )
                lrrr = str(param_group['lr'])

            batch_time.update(time.time() - end)
            losses.update(loss.item(), trainY.size(0))
            htri_losses.update(htri_loss.item(), trainY.size(0))
            accs.update(accuracy(outputs[0:cfg.SOLVER.TRAIN_BATCH_SIZE], trainY[0:cfg.SOLVER.TRAIN_BATCH_SIZE])[0])

            if (batch_idx) % cfg.MISC.PRINT_FREQ == 0:
                print('Train ', end=" ")
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'lr {lrrr} \t'.format(
                    epoch + 1, batch_idx + 1, len(trainloader),
                    batch_time=batch_time,
                    loss=losses,
                    acc=accs,
                    lrrr=lrrr,
                ))

            end = time.time()

        scheduler.step()
        print('=> Test')

        for name in cfg.DATASET.TARGET_NAME:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']

            if gcn_model is not None:
                rank1, distmat = do_test(model=model, queryloader=queryloader, galleryloader=galleryloader, batch_size=cfg.TEST.TEST_BATCH_SIZE, use_gpu=use_gpu, dataset=cfg.DATASET.TARGET_NAME[0], reranking=cfg.TEST.RE_RANKING, graph_reranking=cfg.TEST.GRAPH_RE_RANKING, learn_based=cfg.TEST.LEARN_BASED, gcn_model=gcn_model)
            else:
                rank1, distmat = do_test(model=model, queryloader=queryloader, galleryloader=galleryloader, batch_size=cfg.TEST.TEST_BATCH_SIZE, use_gpu=use_gpu, dataset=cfg.DATASET.TARGET_NAME[0], reranking=cfg.TEST.RE_RANKING, graph_reranking=cfg.TEST.GRAPH_RE_RANKING)

            ranklogger.write(name, epoch + 1, rank1)
            # ranklogger.write(name, epoch + 1, rank2)
            
            if cfg.SOLVER.EARLY_STOPPING:
                if rank1 > best_rank1 + cfg.SOLVER.MIN_DELTA:
                    best_rank1 = rank1
                    best_epoch = epoch + 1
                    patience_counter = 0
                    print(f'[Early Stopping] New best Rank-1: {best_rank1:.2%} at epoch {best_epoch}')
                    # Save best model
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'rank1': rank1,
                        'epoch': epoch + 1,
                        'arch': cfg.MODEL.ARCH,
                        'optimizer': optimizer.state_dict(),
                    }, cfg.MISC.SAVE_DIR, 'best_model')
                    
                    if gcn_model is not None:
                        gcn_best_path = osp.join(cfg.MISC.SAVE_DIR, 'gcn_best_model.pth')
                        torch.save({
                            'state_dict': gcn_model.state_dict(),
                            'epoch': epoch + 1,
                            'rank1': rank1,
                        }, gcn_best_path)
                        print(f'Best GCN model saved to {gcn_best_path}')
                else:
                    patience_counter += 1
                    print(f'[Early Stopping] No improvement. Patience: {patience_counter}/{cfg.SOLVER.PATIENCE}')
                    
                    if patience_counter >= cfg.SOLVER.PATIENCE:
                        print(f'[Early Stopping] Training stopped at epoch {epoch + 1}')
                        print(f'Best Rank-1: {best_rank1:.2%} at epoch {best_epoch}')
                        return
            
            if (epoch + 1) == cfg.SOLVER.MAX_EPOCHS and cfg.TEST.VIS_RANK == True:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(cfg.MISC.SAVE_DIR, 'ranked_results', name),
                    topk=20)

        del queryloader
        del galleryloader
        del distmat
        # print(torch.cuda.memory_allocated(),torch.cuda.memory_cached())
        torch.cuda.empty_cache()

        if (epoch + 1) == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'rank1': rank1,
                'epoch': epoch + 1,
                'arch': cfg.MODEL.ARCH,
                'optimizer': optimizer.state_dict(),
            }, cfg.MISC.SAVE_DIR, cfg.SOLVER.OPTIMIZER_NAME)