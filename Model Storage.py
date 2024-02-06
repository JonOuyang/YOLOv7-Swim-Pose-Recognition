###Training with K-Folds
# Define the number of folds (k)
#model = create_res_net()
#model = tf.keras.models.load_model('testModel65')

for i in range(1):

    k = 10  # For example, using 5-fold cross-validation

    # Create a KFold object
    kf = KFold(n_splits=k, shuffle=True)

    #model.summary()

    # Loop through each fold
    for fold, (train_index, val_index) in enumerate(kf.split(x_test)):
        print(f"Training fold {fold + 1}")
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        """class_weights = {0: 1,
                        1: 10.,
                        2: 10,
                        3: 1}"""
        """ResNet
        class_weights = {0: 0.6,
                         1: 1.2,
                         2: 1.4,
                         3: 1.,
                         4: 1.2,
                         5: 0.9,}"""
        
        model_checkpoint = ModelCheckpoint(filepath = f'model_{fold}.h5', monitor='val_loss')

        def relu_bn(inputs: Tensor) -> Tensor:
    
    relu = layers.LeakyReLU(alpha=0.01)(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    y = Dropout(0.1)(y)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(32, 12, 3))
    num_filters = 100
    
    t = BatchNormalization()(inputs)
    #k_s, 4, 4, stride 2
    t = Conv2D(kernel_size=4,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    kernel_regularizer=l2(0.001)
    t = relu_bn(t)
    
    #2, 4, 4, 2
    #num_blocks_list = [2, 5, 5, 2]
    #enhanced: 5, 9, 9, 5
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters*=2
    
    t = AveragePooling2D(2)(t)
    #t = MaxPooling2D(pool_size=(2, 2))(t)
    t = Flatten()(t)
    #p = Dense(32, activation='relu')(t)
    outputs = Dense(2, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        #optimizer = Adam(learning_rate=0.0001),
        #optimizer = Adam(learning_rate=4e-7),
        optimizer=Adam(learning_rate=3e-6),
        #optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model
model = create_res_net()
        # Train the model on this fold
        history = model.fit(
            x=x_train_fold,
            y=y_train_fold,
            epochs=20,
            verbose=0,
            batch_size=32,
            validation_data=(x_val_fold, y_val_fold),
            #class_weight=class_weights
            #validation_data=(x_test, y_test)
            callbacks=[model_checkpoint]
        )
        model.evaluate(x_test, y_test, verbose=1)
        predicted_labels=np.argmax(model.predict(xt), axis=1)
        tl = np.argmax(yt, axis=1)
        #tl = ytR
        confusion_mat = confusion_matrix(tl, predicted_labels)
        print(confusion_mat)
        print(f'Model Accuracy based on testing data: {(np.trace(confusion_mat))/np.sum(confusion_mat)}')
        
        # Store the training and validation results for this fold

    # After all folds are done, you can analyze the performance using the stored lists.
    # For example, you can compute the average and standard deviation of train and validation metrics.

%%time
accuracyArray = []
loops = 0
while True:
    try:
        #Model
        model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.ZeroPadding2D(padding=(5,5)),
        layers.Conv2D(60, (2, 2)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(65, (2, 2)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(70, (4, 4)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(80, (4, 4)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(90, (5, 5)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(100, (5, 5)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Flatten(), layers.Dropout(0.4),
        layers.Dense(2, activation="softmax"),
    ])
        model.compile(loss="categorical_crossentropy", optimizer = Adam(learning_rate=3e-6), metrics="accuracy")

        accuracyArray = []
        for i in range(1):

            k = 10  # For example, using 5-fold cross-validation

            # Create a KFold object
            kf = KFold(n_splits=k, shuffle=True)

            # Initialize lists to store training and validation results for each fold

            #model.summary()

            # Loop through each fold
            for fold, (train_index, val_index) in enumerate(kf.split(x_test)):
                print(f"Training fold {fold + 1}")
                x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                model_checkpoint = ModelCheckpoint(filepath = f'model_{fold}.h5', monitor='val_loss')


                # Train the model on this fold
                history = model.fit(
                    x=x_train_fold,
                    y=y_train_fold,
                    epochs=20,
                    verbose=0,
                    batch_size=32,
                    validation_data=(x_val_fold, y_val_fold),
                    #class_weight=class_weights
                    #validation_data=(x_test, y_test)
                    callbacks=[model_checkpoint]
                )
                model.evaluate(x_test, y_test, verbose=1)
                predicted_labels=np.argmax(model.predict(xt), axis=1)
                tl = np.argmax(yt, axis=1)
                #tl = ytR
                confusion_mat = confusion_matrix(tl, predicted_labels)
                print(confusion_mat)
                print(f'Model Accuracy based on testing data: {(np.trace(confusion_mat))/np.sum(confusion_mat)}')
                accuracyArray.append((np.trace(confusion_mat))/np.sum(confusion_mat))
        loops +=1
    except:
        pass
    if any(num > 0.74 for num in accuracyArray):
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print(f'Model trained a total of {loops} times!')
        print("Training Complete!")
        break
    else:
        clear_output(wait=True)


###BASE RESNET MODEL
def relu_bn(inputs: Tensor) -> Tensor:
    
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    #bn = ReLU()(inputs)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    #y = Dropout(0.4)(y)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(32, 12, 3))
    num_filters = 64
    
    t = BatchNormalization()(inputs)
    #k_s, 4, 4, stride 2
    t = Conv2D(kernel_size=4,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    #kernel_regularizer=l2(0.01)
    #ernel_regularizer=l2(0.001)
    t = relu_bn(t)
    
    #2, 4, 4, 2
    #num_blocks_list = [2, 5, 5, 2]
    #enhanced: 5, 9, 9, 5
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters*=2
    
    t = AveragePooling2D(2)(t)
    #t = MaxPooling2D(pool_size=(2, 2))(t)
    t = Flatten()(t)
    #p = Dense(32, activation='relu')(t)
    outputs = Dense(2, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    """model.compile(
        #optimizer = Adam(learning_rate=0.0001),
        #optimizer = Adam(learning_rate=4e-7),
        optimizer=Adam(learning_rate=1e-7),
        #optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )"""

    model.compile(
      optimizer = Adam(learning_rate=4e-6),
      #optimizer = 'sgd',
      loss='categorical_crossentropy',
      metrics=[tf.keras.metrics.PrecisionAtRecall(recall=0.8)]
    )
    return model

###Experimental test model 1
"""model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.Conv2D(100, (3, 3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Conv2D(120, (3, 3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Conv2D(120, (3, 3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Conv2D(120, (3, 3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Conv2D(100, (3, 3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(), layers.Dropout(0.4),
        layers.Dense(4, activation="softmax"),
    ])"""
###42% accuracy with problematic classes doubled, 64 batch
"""model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.ZeroPadding2D(padding=(2,2)),
        layers.Conv2D(200, (3, 3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Conv2D(200, (3, 3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Conv2D(200, (3, 3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Conv2D(200, (3, 3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(), layers.Dropout(0.4),
        #layers.Dense(16, activation='relu'),
        layers.Dense(4, activation="softmax"),
    ]) -> 42% acc, c0&1 doubled, c64"""
###40% accuracy consistently, problematic classes doubled, 64 batch
"""model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.ZeroPadding2D(padding=(2,2)),
        layers.Conv2D(500, (5, 5), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.05),
        layers.Conv2D(500, (5, 5), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.05),
        layers.Conv2D(500, (5, 5), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(4, activation="softmax"),
    ]) -> 40% acc, c0&1 doubled, c64"""
###60% accuracy on binary classification between free and back, 32 batch
"""model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.ZeroPadding2D(padding=(6,6)),
        layers.Conv2D(200, (5, 5), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(200, (5, 5), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(2, activation="softmax"),
    ]) -> 60% acc, free/back, c32, """
###65% accuracy on binary classification between free and back, 32 batch
model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.ZeroPadding2D(padding=(6,6)),
        layers.Conv2D(200, (6, 6), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(200, (6, 6), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(40, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(2, activation="softmax"),
    ]) #65% peak fld #4

###.compile
#model.compile(loss="categorical_crossentropy", optimizer = 'sgd', metrics=[tf.keras.metrics.AUC()])
#model.compile(loss="categorical_crossentropy", optimizer = 'sgd', metrics=[tf.keras.metrics.Precision()])
#model.compile(loss="categorical_crossentropy", optimizer = 'sgd', metrics=[tf.keras.metrics.Recall()]) <- overfits severely
#model.compile(loss="categorical_crossentropy", optimizer = 'sgd', metrics=[tf.keras.metrics.TruePositives()])
#model.compile(loss="categorical_crossentropy", optimizer = 'sgd', metrics=[tf.keras.metrics.FalsePositives()])

#Working
#model.compile(loss="categorical_crossentropy", optimizer = Adam(learning_rate=3e-5), metrics=[tf.keras.metrics.PrecisionAtRecall(recall=0.8)])

model.compile(loss="categorical_crossentropy", optimizer = Adam(learning_rate=3e-6), metrics="accuracy")
#model.compile(loss="categorical_crossentropy", optimizer = 'sgd', metrics=[tf.keras.metrics.SensitivityAtSpecificity(0.8)])
#model.compile(loss="categorical_crossentropy", optimizer = 'sgd', metrics=[tf.keras.metrics.SpecificityAtSensitivity(0.8)])

###65% backstroke bias
model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.ZeroPadding2D(padding=(6,6)),
        layers.Conv2D(200, (6, 6), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(200, (6, 6), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(95, activation='relu'),
        layers.Dense(80, activation='relu'),
        
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(2, activation="softmax"),
    ]) #65% peak fld #4

###40-65%, unknown why it doesnt work, main part is large dense layer between convs
model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.ZeroPadding2D(padding=(6,6)),
        layers.Conv2D(200, (6, 6), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(150, activation='relu'),
        layers.Conv2D(200, (6, 6), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(95, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dense(80, activation='relu', kernel_regularizer=l2(0.01)),
        
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(2, activation="softmax"),
    ]) 



-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-
###75% no conv, adam(3e-6)
70 SERIES
model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.Dense(50, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Flatten(),
        layers.Dense(2, activation="softmax"),
    ])
*redo model 78?


#Fly vs Breast, utilizing LEAKYRELU, 65% 
[[43  9]
 [25 20]]

73% 80 SERIES
model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.ZeroPadding2D(padding=(5,5)),
        layers.Conv2D(60, (2, 2)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(65, (2, 2)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(70, (4, 4)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(80, (4, 4)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(90, (5, 5)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv2D(100, (5, 5)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Flatten(), layers.Dropout(0.4),
        layers.Dense(2, activation="softmax"),
    ])

76% 90 series

def relu_bn(inputs: Tensor) -> Tensor:
    
    relu = layers.LeakyReLU(alpha=0.01)(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    y = Dropout(0.1)(y)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(32, 12, 3))
    num_filters = 100
    
    t = BatchNormalization()(inputs)
    #k_s, 4, 4, stride 2
    t = Conv2D(kernel_size=4,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    kernel_regularizer=l2(0.001)
    #ernel_regularizer=l2(0.001)
    t = relu_bn(t)
    
    #2, 4, 4, 2
    #num_blocks_list = [2, 5, 5, 2]
    #enhanced: 5, 9, 9, 5
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters*=2
    
    t = AveragePooling2D(2)(t)
    #t = MaxPooling2D(pool_size=(2, 2))(t)
    t = Flatten()(t)
    #p = Dense(32, activation='relu')(t)
    outputs = Dense(2, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        #optimizer = Adam(learning_rate=0.0001),
        #optimizer = Adam(learning_rate=4e-7),
        optimizer=Adam(learning_rate=3e-6),
        #optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model
model = create_res_net()
