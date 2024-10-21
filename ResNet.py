def ResNet34(width, height, depth, classes):
    Img = Input(shape=(width, height, depth))
    
    # Initial Convolution and Pooling
    x = Conv2d_BN(Img, 64, (7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Residual Blocks
    x = Residual_Block(x, 64, (3, 3))
    for _ in range(2):  # Reduce repetition
        x = Residual_Block(x, 64, (3, 3))
    
    x = Residual_Block(x, 128, (3, 3), strides=(2, 2), with_conv_shortcut=True)
    for _ in range(3):
        x = Residual_Block(x, 128, (3, 3))

    x = Residual_Block(x, 256, (3, 3), strides=(2, 2), with_conv_shortcut=True)
    for _ in range(5):
        x = Residual_Block(x, 256, (3, 3))

    x = Residual_Block(x, 512, (3, 3), strides=(2, 2), with_conv_shortcut=True)
    for _ in range(2):
        x = Residual_Block(x, 512, (3, 3))

    # Final Layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)
    
    model = Model(inputs=Img, outputs=x)
    return model
