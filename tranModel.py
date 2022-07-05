import torch
import os
import cv2
import random
from torch.utils.data import DataLoader, Dataset
from torch import nn
import json

if torch.cuda.is_available():
    x = "cuda"
else:
    x = 'cpu'
device = torch.device(x)
print('正在使用'+x+'进行训练')

if not os.path.isdir('./ModelYZM'):
    os.mkdir('./ModelYZM')



class dataOfMe(Dataset):
    def __init__(self,info,lableList,number,lenYzm,fun=None):
        super(dataOfMe,self).__init__()
        self.info = info
        self.lableList = lableList
        self.number = number
        self.fun = fun
        self.lenYzm = lenYzm

    def getLable(self,data):
        dati = []
        # print(self.number)
        for i in data:
            o = [0 for i in range(self.number)]
            index = self.lableList.index(i)
            o[index] = 1
            dati += o
        return dati
    def getLable2(self,data):
        dati = []
        for i in data:
            dati.append(self.lableList.index(i))
        return dati

    def __getitem__(self, item):
        img = cv2.imread('./parise/'+self.info[item])
        if self.fun != None:
            img = self.fun(img)
        target = self.info[item].replace('.png','')
        if len(target) < self.lenYzm:
            target += '@'*(self.lenYzm-len(target))
        target = self.getLable2(target)
        target = torch.tensor(target,dtype=torch.float32).to(device)
        img = torch.tensor(img,dtype=torch.float32).transpose(2,0).transpose(2,1).to(device)
        return img,target

    def __len__(self):
        return len(self.info)


class Tu(nn.Module):
    def __init__(self):
        super(Tu,self).__init__()
        self.lin = nn.Linear(50*200, 5)

    def forward(self,x):
        x = self.cnn1(x)
        x = self.maxPool1(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.lin(x)

        return x


def getInfo(x):
    img = cv2.imread(x)
    img =  torch.tensor(fun(img),dtype=torch.float32).unsqueeze(0)
    img = img.transpose(3,1).transpose(3,2)
    img.to(device)
    return img


def train(batch_size=20,tranL=1,model=None):
    from tqdm import tqdm


    info = []
    info2 = []
    data = os.listdir('./parise')
    random.shuffle(data)
    number = data[200:]
    number2 = data[:200]
    yzmLen = 0
    koyzmFu = '@'

    data = set('')
    for i in number2:
        if i.find('.png') != -1:
            info2.append(i)
            for s in i.replace('.png', ''):
                data.add(s)


    for i in number:
        if i.find('.png') != -1:
            info.append(i)
            if len(i.replace('.png', '')) > yzmLen:
                yzmLen = len(i.replace('.png', ''))
            for s in i.replace('.png', ''):

                data.add(s)

    lableList = list(data)

    lableList = [koyzmFu]+ lableList

    tdao = getInfo('./parise/'+info[0]).shape[1]

    data_train = dataOfMe(info, lableList, len(lableList), yzmLen,fun)
    ff = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    # ee = Tu(len(lableList))
    if model != None:
        ee = model
    else:
        import torchvision
        ee = torchvision.models.resnet18(num_classes=yzmLen)
        ee.conv1 = nn.Conv2d(tdao, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    ee.to(device)
    loss = nn.MSELoss()

    optim = torch.optim.Adam(ee.parameters(), lr=1e-3)
    data_train2 = dataOfMe(info2,lableList,len(lableList), yzmLen,fun)
    ff2 = DataLoader(data_train2, batch_size=batch_size, shuffle=True)

    for i in range(tranL):
        ff = tqdm(ff, total=len(ff), desc="训练数据进度")
        lossMe = 0
        zs = 0
        for k,t in ff:
            out = ee(k)
            optim.zero_grad()
            result_loss = loss(out, t)
            result_loss.backward()
            optim.step()
            zs += 1
            ff.set_postfix(loss=result_loss.item(),cbb=i+1)
            lossMe += result_loss.item()
        print()
        print("\r平均损失："+str(lossMe/zs))
        zq = 0
        yg = 0
        ff2 = tqdm(ff2, total=len(ff2), desc="测试数据进度")
        for k,t in ff2:
            if t.shape[0] != 20:
                continue
            out = ee(k)
            s = out-t
            s = torch.abs(s)
            s = s.sum()
            zq += s.item()
            yg += yzmLen
        print(end='\r')
        print("平均准确率损失: ", zq/yg)
    torch.save(ee,'./ModelYZM/modelyzm.pth')
    with open('./ModelYZM/listLable.conf','w',encoding='utf-8') as f:
        f.write(json.dumps(lableList))

def yuche(data):
    with open('./ModelYZM/listLable.conf',encoding='utf-8') as f:
        lableList = json.loads(f.read())
    ee = torch.load('./ModelYZM/model_name.pth')
    ee.to(device)
    y = getInfo(data)
    u = ee(y)
    lab = torch.round(u)

    print(lableList)
    return ''.join([lableList[int(i)] for i in list(lab.data.view(-1))])


# 对取求的图片进行处理
def fun(img):
    return img


if __name__ == '__main__':

    # 训练模型
     # batch_size设置一次取几张图片进行训练
    batch_size = 20
     # tranL设置训练多少轮
    tranL = 10
     # model设置自己模型进行训练
    train(batch_size,tranL, model=None)

    # 下面是识别图片验证码的代码代码
    # yzm = yuche("./parise/gabc.png")
    # print(yzm)