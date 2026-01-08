- [ ] ARMAZENAR O TEMPO DE CADA EPOCA NO TREINAMENTO (COMPARAR COM O TENSORFLOW O TEMPO TOTAL OU O TEMPO MÉDIO)
- [ ] APLICAR DROPOUT NAS CAMADAS TOTALMENTE CONECTADAS (TENSORFLOW TEM DROPOUT)
- [ ] SALVAR A ARQUITETURA E PARAMETROS DA ARQUITETURA DE CNN PARA USAR DEPOIS APENAS NO TREINAMENTO
- [ ] VERIFICAR SE COMPENSA SALVAR O LOSS E MAE PARA COMPARAR COM O TENSORFLOW PARA CADA ÉPOCA
- [ ] VERIFICAR ALGUNS PARÂMETROS QUE EXISTEM NO TENSORFLOW (VALIDATION_SPLIT, BATCH_SIZE)
- [ ] SALVAR ACURÁCIA DO TESTE (TANTO NO GERAL QUANTO POR CLASSE)



----------------------------------------------------------------------
<!-- # def CalcMeanAndStd():
    #     trainSet, testSet = DefineTrainSetTestSet()
    #     imgs = [item[0] for item in trainSet] # item[0] imagens e item[1] classe
    #     imgs = torch.stack(imgs, dim=0).numpy()

    #     mean_r = imgs[:,0,:,:].mean()
    #     mean_g = imgs[:,1,:,:].mean()
    #     mean_b = imgs[:,2,:,:].mean()
    #     print(mean_r,mean_g,mean_b)

    #     std_r = imgs[:,0,:,:].std()
    #     std_g = imgs[:,1,:,:].std()
    #     std_b = imgs[:,2,:,:].std()
    #     print(std_r,std_g,std_b) -->