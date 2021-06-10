import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from config import *
from draw_model import DrawModel
from utility import Variable, save_image, xrecons_grid
from dataset import Font

torch.set_default_tensor_type('torch.FloatTensor')

# dataset = datasets.MNIST('data/', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor()]))
dataset = Font('D:/code/python/manga/result')

# image channel value range = [0, 1]
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4)

model = DrawModel(T,A,B,z_size,N,dec_size,enc_size)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,betas=(beta1,0.999))

if USE_CUDA:
    model.cuda()

count_thres = 100
def train():
    avg_loss = 0
    count = 0
    for epoch in range(epoch_num):
        for data, label in train_loader:
            bs = data.size()[0]
            data = Variable(data).view(bs, -1)
            if USE_CUDA:
                data = data.cuda()
            optimizer.zero_grad()
            loss = model.loss(data)
            avg_loss += loss.cpu().data.numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

            count += 1
            if count % count_thres == 0:
                print('Epoch-{}; Count-{}; loss: {};'.format(epoch, count, avg_loss / count_thres))
                avg_loss = 0

            if count % 100 == 0:
                torch.save(model.state_dict(),'save/weights_%d.tar'%(count))
                generate_image(count)
        # torch.save(model.state_dict(),'save/weights_epoch_%d.tar'%(epoch))
        # generate_image(epoch)
    torch.save(model.state_dict(), 'save/weights_final.tar')
    generate_image(count)

def generate_image(count):
    x = model.generate(batch_size)
    save_image(x, count)

def save_example_image():
    train_iter = iter(train_loader)
    data, _ = train_iter.next()
    img = data.cpu().numpy().reshape(batch_size, 28, 28)
    imgs = xrecons_grid(img, B, A)
    plt.matshow(imgs, cmap=plt.cm.gray)
    plt.savefig('image/example.png')

if __name__ == '__main__':
    # for data, label in train_loader:
    #     print(data)
    #     print(data.shape)
    #     print(label)
    print(f'Train size : {len(dataset)}')
    save_example_image()
    train()
