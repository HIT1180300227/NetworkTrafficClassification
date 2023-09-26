# 尝试使用text_cnn--一维卷积神经网络处理图像分类问题
# 针对FlowPic一文中提出的方法，其生成的图像可以认为是一段时间内的平移
# 而在这样的n*m的矩形图像中，纵向具有可区分度
# 尝试使用nlp任务中的text_cnn来进行此类图像的分类
# 前置工作见cnn.py，text_cnn思想用于图像分类有待验证
import copy
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

#定义text_cnn网络：
class MyTextCnn(nn.Module):

    def __init__(self, dim, n_filter, filter_size, out_dim):

        super(MyTextCnn,self).__init__()

        #卷积层
        self.cov=nn.ModuleList([
            nn.Conv1d(in_channels=1,
                      out_channels=n_filter,
                      stride=(fs,),
                      kernel_size=(fs,dim)) for fs in filter_size

        ])

        #第二层卷积
        self.cov2 = nn.ModuleList([
            nn.Conv1d(in_channels=n_filter,
                      out_channels=2*n_filter,
                      stride=(fs,),
                      kernel_size=(fs,1)) for fs in filter_size

        ])

        #全连接层：
        self.fc=nn.Linear(len(filter_size)*2*n_filter*3,out_dim)

    def k_max_pool(self,x,dim,k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(self,x):

        #第一层卷积
        #coved = [torch.relu(cov(x)).squeeze(3) for cov in self.cov]
        coved = [torch.relu(cov(x)) for cov in self.cov]

        #第二层卷积
        coved_twice=[]
        for i,x in enumerate(coved):
            coved_twice.append(torch.relu(self.cov2[i](x)))

        #1-Max池化
        #pooled = [torch.max_pool1d(cov,cov.shape[2]).squeeze(2) for cov in coved]
        #pooled=[torch.max_pool1d(cov.squeeze(3),kernel_size=(cov.shape[2],)) for cov in coved_twice]
        #k-Max池化
        pooled=[self.k_max_pool(cov,dim=2,k=3) for cov in coved_twice]

        #cat = self.dropout(torch.cat(pooled,dim=1))
        cat = torch.cat(pooled,dim=1)
        cat = torch.flatten(cat, 1)

        return self.fc(cat)


#定义训练函数：以validation集上的正确率作为选择标准
def trainmodel(model,traindataloader,train_rate,criterion,optimizer,num_epochs):

    #计算训练用的batch数量
    batch_num=len(traindataloader)

    train_batch_num=round(batch_num*train_rate)

    #复制模型的参数
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.0

    #测试
    train_loss_all=[]
    train_acc_all=[]

    #验证
    val_loss_all=[]
    val_acc_all=[]

    since=time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)

        train_loss=0.0
        train_correct=0
        train_num=0
        val_loss = 0.0
        val_correct = 0
        val_num = 0


        for step,(b_x,b_y) in enumerate(traindataloader):

            # 划分train和validate
            if step < train_batch_num:
                #此步存疑,可能是出错的原因
                model.train()

                output=model(b_x)
                loss = criterion(output, b_y)

                prelab=torch.argmax(output,1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #统计：
                # print(loss.item())
                # print(b_x.size(0))
                train_loss+=loss.item()*b_x.size(0)
                #train_correct+=torch.eq(prelab, b_y.data).sum()
                train_correct+=torch.sum(prelab == b_y.data)
                train_num+=b_x.size(0)

            else:
                #验证阶段，相当于测试
                model.eval()
                output=model(b_x)
                prelab=torch.argmax(output,1)
                print(prelab)

                loss=criterion(output,b_y)
                val_loss += loss.item() * b_x.size(0)
                #val_correct += torch.eq(prelab, b_y.data).sum()
                val_correct += torch.sum(prelab == b_y.data)
                val_num += b_x.size(0)

        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_correct/train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_correct / val_num)


        print('{} train loss : {:.4f},train acc : {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} val loss : {:.4f},val acc : {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        #保存最佳参数
        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())
        timeuse=time.time()-since
        print('train and val complete in {:.0f}m {:.0f}s'.format(timeuse//60,timeuse%60))

    model.load_state_dict(best_model_wts)

    return model

#以loss作为参数选择依据，取消validation集
def trainmodel2(model,traindataloader,train_rate,criterion,optimizer,num_epochs):

    #计算训练用的batch数量
    batch_num=len(traindataloader)

    train_batch_num=round(batch_num*train_rate)

    #复制模型的参数
    best_model_wts=copy.deepcopy(model.state_dict())
    best_loss=100

    #测试
    train_loss_all=[]
    train_acc_all=[]

    #验证
    val_loss_all=[]
    val_acc_all=[]

    since=time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)

        train_loss=0.0
        train_correct=0
        train_num=0
        val_loss = 0.0
        val_correct = 0
        val_num = 0


        for step,(b_x,b_y) in enumerate(traindataloader):

            # 划分train和validate
            if step < train_batch_num:

                model.train()

                output=model(b_x)
                loss = criterion(output, b_y)
                prelab=torch.argmax(output,1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #统计：
                # print(loss.item())
                # print(b_x.size(0))
                train_loss+=loss.item()*b_x.size(0)
                #train_correct+=torch.eq(prelab, b_y.data).sum()
                train_correct+=torch.sum(prelab == b_y.data)
                train_num+=b_x.size(0)


            else:
                #验证阶段，相当于测试集-Test
                model.eval()
                output=model(b_x)
                prelab=torch.argmax(output,1)
                print(prelab)
                loss=criterion(output,b_y)
                val_loss += loss.item() * b_x.size(0)
                #val_correct += torch.eq(prelab, b_y.data).sum()
                val_correct += torch.sum(prelab == b_y.data)
                val_num += b_x.size(0)

        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_correct/train_num)
        #val_loss_all.append(val_loss / val_num)
        #val_acc_all.append(val_correct / val_num)


        print('{} train loss : {:.4f},train acc : {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        #print('{} val loss : {:.4f},val acc : {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        #保存最佳参数
        if train_loss_all[-1]<best_loss:
            best_loss=train_loss_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())
        timeuse=time.time()-since
        print('train and val complete in {:.0f}m {:.0f}s'.format(timeuse//60,timeuse%60))

    model.load_state_dict(best_model_wts)

    return model

#使用pytorch必须
if __name__ == '__main__':

    # 加载数据,使用ImageFolder，读取灰度图片，transforms.Grayscale(num_output_channels=1),
    train_data_transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]
    )

    #根目录
    train_data_dir = ''

    #data
    train_data = ImageFolder(train_data_dir, transform=train_data_transform)

    #loader
    train_data_loader = Data.DataLoader(train_data, batch_size=20, shuffle=True, num_workers=4)

    classlabel = train_data.classes

    #定义神经网络：
    mynet = MyTextCnn(100,40,[4,5,6],10)

    #训练模型：
    optimizer=torch.optim.Adam(mynet.parameters())

    #损失函数：
    critierion=nn.CrossEntropyLoss()

    #训练数据
    mynet=trainmodel2(mynet,train_data_loader,1,critierion,optimizer,20)

    #保存模型
    torch.save(mynet, 'model_100px/model11.pkl')

