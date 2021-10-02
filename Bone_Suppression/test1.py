# def getModel():
#     model = AE()
#     learning_rate = 1e-3
#     criterion = nn.MSELoss()
#     optimizer = Adam(model.parameters(), 
#     lr=learning_rate, weight_decay=1e-5)

#     return model, criterion, optimizer

# def train(path, trained=False):
    
#     model, criterion, optimizer = getModel()
#     num_epochs = 50

#     total_loss = 0
#     for epoch in range(num_epochs):
#         for data in dataloader:
#             img, _ = data
#             img = Variable(img).cuda()
#             # ===================forward=====================
#             output = model(img)
#             loss = criterion(output, img)
#             # ===================backward====================
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         # ===================log========================
#         total_loss += loss.data
#         print('epoch [{}/{}], loss:{:.4f}'
#             .format(epoch+1, num_epochs, total_loss))
#         if epoch % 10 == 0:
#             pic = to_img(output.cpu().data)
#             save_image(pic, './dc_img/image_{}.png'.format(epoch))

    # torch.save(model.state_dict(), './conv_autoencoder.pth')