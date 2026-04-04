from __future__ import print_function
from __future__ import division
import time
import numpy as np
import torch
from utils.evaluation import evaluate, evaluate_vid
from utils.reranking import re_ranking
from utils.avgmeter import AverageMeter
from tqdm import tqdm
from utils.graph_reranking import *


def do_test(model, queryloader, galleryloader, batch_size, use_gpu, dataset, ranks=[1, 5, 10], reranking=False, graph_reranking=False, learn_based=False, gcn_model=None):
    batch_time = AverageMeter()

    model.eval()
    if gcn_model is not None:
        gcn_model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(tqdm(queryloader)):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(tqdm(galleryloader)):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, batch_size))

    if reranking:
        print("Start Original Re-ranking \n")
        distmat = re_ranking(qf, gf, k1=80, k2=15, lambda_value=0.2)
        print('Computing CMC and mAP')
        if dataset == 'vehicleid':
            cmc, mAP = evaluate_vid(distmat, q_pids, g_pids, q_camids, g_camids, 50)
        else:
            cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, 50)
        print('Re-Ranked Results--')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
        print('------------------')
    elif graph_reranking:
        if learn_based and gcn_model is not None:
            print("Start GCN Model for Graph Re-ranking \n")
            distmat = graph_reranking_func(qf, gf, q_camids, g_camids, gcn_model=gcn_model)
            print('Computing CMC and mAP')
            if dataset == 'vehicleid':
                cmc, mAP = evaluate_vid(distmat, q_pids, g_pids, q_camids, g_camids, 50)
            else:
                cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, 50)
            print('GCN Model for Graph Re-ranked Results--')
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
            print('------------------')
        else:
            print("Start Graph Re-Ranking \n")
            distmat = graph_reranking_func(qf, gf, q_camids, g_camids)
            print('Computing CMC and mAP')
            if dataset == 'vehicleid':
                cmc, mAP = evaluate_vid(distmat, q_pids, g_pids, q_camids, g_camids, 50)
            else:
                cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, 50)
            print('Graph Re-ranked Results--')
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
            print('------------------')
    else:
        print("No Re-ranking \n")
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        distmat = distmat.numpy()

        print('Computing CMC and mAP')
        if dataset == 'vehicleid':
            cmc, mAP = evaluate_vid(distmat, q_pids, g_pids, q_camids, g_camids, 50)
        else:
            cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, 50)

        print('No Re-ranked Results ----------')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
        print('------------------')


    return cmc[0], distmat