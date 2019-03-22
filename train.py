# coding: utf-8
from utils import *
from model import *
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

EPOCHS = 1000
HASH_BITS = [48]
LOSS_01 = 0.001
LR = 0.001
CHECKPOINT_PATH = 'checkpoints'
HAMMING_DISTANCE = 2
m = 0


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.5 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def criteria_softmax(hash_bits, out, target, beta):
    target = target.view(-1)
    softmax_loss = F.nll_loss(out, target).sum()
    quatization_loss = beta * torch.sum((hash_bits.abs() - 1).abs(), dim=1).mean()
    return softmax_loss + quatization_loss


if __name__ == '__main__':
    # load data
    train_loader, test_loader = get_loader()
    # get the device , use gpu when it's availiable.
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("using: " + str(device))
    test_maps = []
    for bits in HASH_BITS:
        net = DDAH(bits, C).to(device)
        optimizer_a = optim.Adam([{'params': net.spatial_features_1.parameters()},
                                  {'params': net.spatial_features_2.parameters()},
                                  {'params': net.spatial_features_3.parameters()},
                                  {'params': net.fc.parameters()},
                                  {'params': net.upscales_1.parameters()},
                                  {'params': net.upscales_2.parameters()},
                                  {'params': net.upscales_3.parameters()},
                                  {'params': net.upscales_4.parameters()}], lr=LR, weight_decay= 0.0001)

        optimizer_b = optim.Adam([{'params': net.features.parameters()},
                                  {'params': net.conv4.parameters()},
                                  {'params': net.global_avgpooling_3.parameters()},
                                  {'params': net.global_avgpooling_4.parameters()},
                                  {'params': net.face_features_layer.parameters()},
                                  {'params': net.hash_layer.parameters()},
                                  {'params': net.classifier.parameters()}], lr=LR, weight_decay= 0.0001)
        # train start!
        print("bits: " + str(bits))
        for epoch in range(EPOCHS):
            start_time = datetime.now()
            lr = adjust_learning_rate(optimizer_a, epoch)
            lr = adjust_learning_rate(optimizer_b, epoch)

            accum_loss = 0.0
            for batch, (imgs, labels) in enumerate(train_loader):
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer_a.zero_grad()
                optimizer_b.zero_grad()

                hash_bits_a, out_a, hash_bits_b, out_b = net(imgs)
                loss_a = criteria_softmax(hash_bits_a, out_a, labels, LOSS_01)
                loss_b = criteria_softmax(hash_bits_b, out_b, labels, LOSS_01)
                loss_c = loss_a + loss_b
                accum_loss += loss_c.data
                loss_c.backward(retain_graph=True)
                optimizer_b.step()
                optimizer_b.zero_grad()
                #m = loss_a.detach() / 16
                loss_s = torch.clamp(m - loss_b + loss_a, 0)
                loss_s.backward()
                optimizer_a.step()

            trn_binary, trn_label = compute_result(train_loader, net, device)
            tst_binary, tst_label = compute_result(test_loader, net, device)
            mAP = compute_mAP(trn_binary, tst_binary, trn_label, tst_label, device)
            end_time = datetime.now()
            print("[epoch: %d]\t[loss: %.5f]\t[mAP: %.5f]\t[lr: %.5f]\t[time: %d s]" % (epoch+1, accum_loss.data, mAP, lr ,(end_time-start_time).seconds))
       
        trn_binary, trn_label = compute_result(train_loader, net, device)
        tst_binary, tst_label = compute_result(test_loader, net, device)
        mAP = compute_mAP(trn_binary, tst_binary, trn_label, tst_label, device)
        print("[test][bit: %d][mAP: %.5f]" % (bits, mAP) )
        test_maps.append(mAP)
        torch.save(net.state_dict(), CHECKPOINT_PATH + "/" + "facescrub-"+str(bits))
    print(test_maps)
