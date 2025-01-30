expection = hypothesis.view(hypothesis.size(0), hypothesis.size(2), -1)
    # expection = F.softmax(expection, dim=1)
    # expection = expection.argmax(dim=2)
    # accuracy = (expection == y_train).float().mean()

    # if epoch % (epochs / 10) == 0 :
    #     print('Epoch {:4d}/{}, Cost: {:.6f}, Accuracy: {}'.format(epoch, epochs, cost.item(), accuracy))    