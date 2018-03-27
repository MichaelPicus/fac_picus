def jingbai_ds(request):
    lasso_model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_jingbai.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/jingbai_ready.csv'))

    train_y = df_ready.M.values

    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = lasso_model.predict(train)

    combine = np.column_stack((np.expm1(train_pred), train))
    
    # need to delete the origin upload file
    os.remove(os.path.join(BASE_DIR, 'media/documents/jingbai_ready.csv'))

    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
       
        if combine[x, 0] > 33 :
            # AirOutTemp
            modified_res[x, 1] = combine[x, 1]  * 0.98
            # BasePowderTemp
            modified_res[x, 2] = combine[x, 2]  * 0.99123
            # AirInTemp_1
            modified_res[x, 3] = combine[x, 3]  * 1.0456
            # SlurryTemp
            modified_res[x, 4] = combine[x, 4]  * 1.0678

            # TowerTopNegativePressure
            modified_res[x, 5] = combine[x, 5] * 0.9877
            # AgingTankFlow
            modified_res[x, 6] = combine[x, 6] * 1.09678
            # SecondInputAirTemp
            modified_res[x, 7] = combine[x, 7] * 1.00234
            # SlurryPipelineLowerLayerPressure
            modified_res[x, 8] = combine[x, 8] * 1.0285
            # OutAirMotorFreq
            modified_res[x, 9] = combine[x, 9] * 0.9645
            # SecondAirMotorFreq
            modified_res[x, 10] = combine[x, 10] * 0.98235
            # HighPressurePumpFreq
            modified_res[x, 11] = combine[x, 11] * 1.054
            # GasFlow
            modified_res[x, 12] = combine[x, 12] * 0.98667

            modified_res[x, 0] = np.expm1(lasso_model.predict(np.reshape(modified_res[x][1:], (-1, 12))))
            

    final_com = np.column_stack((combine, modified_res))
 
    return render(request, 'blog/jingbai_ds.html', {'final_com': final_com})