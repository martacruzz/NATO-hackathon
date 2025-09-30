# NATO-hackathon


## output classification_pedro:
181/181 ━━━━━━━━━━━━━━━━━━━━ 115s 634ms/step - accuracy: 0.9829 - loss: 0.0615 - precision: 0.9854 - recall: 0.9809 - val_accuracy: 0.9866 - val_loss: 0.0430 - val_precision: 0.9873 - val_recall: 0.9862 Model built: True Number of layers: 12 Model: "sequential" ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓ ┃ Layer (type) ┃ Output Shape ┃ Param # ┃ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩ │ conv2d (Conv2D) │ (None, 128, 128, 8) │ 80 │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ batch_normalization │ (None, 128, 128, 8) │ 32 │ │ (BatchNormalization) │ │ │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ conv2d_1 (Conv2D) │ (None, 64, 64, 16) │ 1,168 │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ batch_normalization_1 │ (None, 64, 64, 16) │ 64 │ │ (BatchNormalization) │ │ │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ conv2d_2 (Conv2D) │ (None, 32, 32, 32) │ 4,640 │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ batch_normalization_2 │ (None, 32, 32, 32) │ 128 │ │ (BatchNormalization) │ │ │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ conv2d_3 (Conv2D) │ (None, 16, 16, 64) │ 18,496 │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ batch_normalization_3 │ (None, 16, 16, 64) │ 256 │ │ (BatchNormalization) │ │ │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ global_average_pooling2d │ (None, 64) │ 0 │ │ (GlobalAveragePooling2D) │ │ │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ embedding_layer (Dense) │ (None, 64) │ 4,160 │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ dropout (Dropout) │ (None, 64) │ 0 │ ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤ │ dense (Dense) │ (None, 24) │ 1,560 │ └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘ Total params: 91,274 (356.54 KB) Trainable params: 30,344 (118.53 KB) Non-trainable params: 240 (960.00 B) Optimizer params: 60,690 (237.07 KB) WARNING:absl:The `save_format` argument is deprecated in Keras 3. We recommend removing this argument as it can be inferred from the file path. Received: save_format=keras Model saved as 'final_drone_classifier.keras' 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 151ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step (There are like a 1000 more of these lines) 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step Class centers and thresholds saved



## Output Labeling YOLO:
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/20      2.44G    0.01964    0.07302     0.8962          4        640: 100% ━━━━━━━━━━━━ 725/725 5.8it/s 2:04
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 91/91 4.1it/s 22.0s
                   all       2896       2896      0.994      0.991      0.995      0.995

20 epochs completed in 0.841 hours.
Optimizer stripped from /data/NATO-hackathon/labeling/drone_detection/drone_detection/weights/last.pt, 6.3MB
Optimizer stripped from /data/NATO-hackathon/labeling/drone_detection/drone_detection/weights/best.pt, 6.3MB

Validating /data/NATO-hackathon/labeling/drone_detection/drone_detection/weights/best.pt...
Ultralytics 8.3.203 🚀 Python-3.12.3 torch-2.8.0+cu128 CUDA:0 (NVIDIA RTX 4000 SFF Ada Generation, 20055MiB)
[W930 10:10:34.513118533 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.515367476 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.515592287 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.515862371 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.516068913 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.516400806 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.516546178 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.516725769 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.516863581 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.517060452 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.517216634 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.517371966 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.517533668 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.517675499 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.518078293 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.518224705 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.518392456 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.518572198 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.518746760 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.518935432 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.519105193 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.519276885 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.519418616 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.519557518 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.519708440 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.519844181 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.522417616 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.523450237 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.523642259 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.523807501 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.524014043 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.524259586 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.524442867 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.524591578 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.524752301 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.524891052 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.525108024 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.525269765 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.525453607 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.525640100 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.525805621 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.525991353 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.528157875 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.528301476 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.528456777 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.528610960 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.528793941 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.528959023 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.529174885 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.529363917 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.529514019 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.529723561 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.529963953 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.530147405 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.530326037 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.530570039 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.530744842 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.530932713 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.534112425 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.534236926 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.534423588 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.534569340 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.534711561 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:10:34.535523639 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
Model summary (fused): 72 layers, 3,010,328 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 91/91 3.2it/s 28.8s
                   all       2896       2896      0.997      0.998      0.995      0.995
Background, including WiFi and Bluetooth        141        141      0.998          1      0.995      0.995
         DJI Phantom 3        138        138      0.999          1      0.995      0.995
     DJI Phantom 4 Rro        152        152      0.994          1      0.995      0.995
       DJI MATRICE 200        151        151      0.992      0.987      0.995      0.995
       DJI MATRICE 100        140        140      0.999          1      0.995      0.995
            DJI Air 2S        134        134      0.999          1      0.995      0.995
        DJI Mini 3 Pro        148        148      0.998          1      0.995      0.995
         DJI Inspire 2        133        133      0.992       0.98      0.994      0.994
         DJI Mavic Pro         67         67      0.998          1      0.995      0.995
            DJI Mini 2        152        152      0.999          1      0.995      0.995
           DJI Mavic 3        109        109      0.998          1      0.995      0.995
       DJI MATRICE 300         59         59      0.998          1      0.995      0.995
 DJI Phantom 4 Pro RTK        107        107      0.999          1      0.995      0.995
       DJI MATRICE 30T         80         80      0.999          1      0.995      0.995
             DJI AVATA        132        132          1      0.987      0.995      0.995
               DJI DIY        107        107      0.987          1      0.994      0.994
   DJI MATRICE 600 Pro        110        110          1      0.993      0.995      0.995
                  VBar        110        110      0.991          1      0.995      0.995
             FrSky X20        124        124      0.999          1      0.995      0.995
          Futaba T16IZ        116        116      0.995          1      0.995      0.995
          Taranis Plus        129        129      0.999          1      0.995      0.995
        RadioLink AT9S        123        123      0.999          1      0.995      0.995
          Futaba T14SG        102        102      0.999          1      0.995      0.995
              Skydroid        132        132      0.999          1      0.995      0.995
Speed: 0.2ms preprocess, 4.1ms inference, 0.0ms loss, 1.9ms postprocess per image
Results saved to /data/NATO-hackathon/labeling/drone_detection/drone_detection

✅ Training complete! Best model saved in runs/detect/drone_detection/weights/best.pt

📊 Evaluating model performance...
Ultralytics 8.3.203 🚀 Python-3.12.3 torch-2.8.0+cu128 CUDA:0 (NVIDIA RTX 4000 SFF Ada Generation, 20055MiB)
[W930 10:11:06.099406089 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.099697733 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.099855654 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.100024786 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.100158447 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.100307208 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.100446891 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.100585422 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.100699833 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.100825314 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.100951805 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.101091546 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.101228178 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.101337799 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.101534211 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.101648252 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.101787003 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.101929246 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.102084027 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.102247478 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.102400940 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.102606102 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.102749464 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.102909495 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.103102237 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.103251768 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.103525042 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.103785934 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.103959605 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.104135108 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.104298429 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.104507611 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.104640952 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.104763363 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.104895876 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.105022477 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.105194628 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.105333379 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.105498431 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.105657783 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.105788814 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.105957596 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.106132857 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.106289089 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.106466031 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.106635413 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.106790924 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.106937465 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.107136338 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.107285069 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.107440531 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.107615112 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.107816584 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.107961366 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.108143038 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.108342270 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.108499621 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.108667014 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.108824455 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.108936946 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.109119318 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.109270409 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.109394901 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
[W930 10:11:06.109872185 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
Model summary (fused): 72 layers, 3,010,328 parameters, 0 gradients, 8.1 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3521.5±460.4 MB/s, size: 679.8 KB)
val: Scanning /data/NATO-hackathon/yolo_dataset/labels/val.cache... 2896 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 2896/2896 35.9Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 181/181 5.3it/s 34.4s
                   all       2896       2896      0.997      0.998      0.995      0.995
Background, including WiFi and Bluetooth        141        141      0.998          1      0.995      0.995
         DJI Phantom 3        138        138      0.999          1      0.995      0.995
     DJI Phantom 4 Rro        152        152      0.995          1      0.995      0.995
       DJI MATRICE 200        151        151      0.992      0.987      0.995      0.995
       DJI MATRICE 100        140        140      0.999          1      0.995      0.995
            DJI Air 2S        134        134      0.999          1      0.995      0.995
        DJI Mini 3 Pro        148        148      0.998          1      0.995      0.995
         DJI Inspire 2        133        133       0.99      0.977      0.994      0.994
         DJI Mavic Pro         67         67      0.998          1      0.995      0.995
            DJI Mini 2        152        152      0.999          1      0.995      0.995
           DJI Mavic 3        109        109      0.998          1      0.995      0.995
       DJI MATRICE 300         59         59      0.998          1      0.995      0.995
 DJI Phantom 4 Pro RTK        107        107      0.999          1      0.995      0.995
       DJI MATRICE 30T         80         80      0.999          1      0.995      0.995
             DJI AVATA        132        132          1      0.987      0.995      0.995
               DJI DIY        107        107      0.988          1      0.995      0.995
   DJI MATRICE 600 Pro        110        110          1      0.993      0.995      0.995
                  VBar        110        110      0.991          1      0.995      0.995
             FrSky X20        124        124      0.999          1      0.995      0.995
          Futaba T16IZ        116        116      0.995          1      0.995      0.995
          Taranis Plus        129        129      0.999          1      0.995      0.995
        RadioLink AT9S        123        123      0.999          1      0.995      0.995
          Futaba T14SG        102        102      0.999          1      0.995      0.995
              Skydroid        132        132      0.999          1      0.995      0.995
Speed: 0.7ms preprocess, 5.2ms inference, 0.0ms loss, 1.6ms postprocess per image
Results saved to /data/NATO-hackathon/labeling/drone_detection/drone_detection2

📊 Overall Metrics:
mAP@0.5: 0.9949
Traceback (most recent call last):
  File "/data/NATO-hackathon/labeling/train_yolo.py", line 35, in <module>
    print(f"mAP@0.5:0.95: {results.box.map50_95:.4f}")
                           ^^^^^^^^^^^^^^^^^^^^
  File "/data/NATO-hackathon/labeling/venv/lib/python3.12/site-packages/ultralytics/utils/__init__.py", line 274, in __getattr__
    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")
AttributeError: 'Metric' object has no attribute 'map50_95'. See valid attributes below.

    Class for computing evaluation metrics for Ultralytics YOLO models.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50: AP at IoU threshold of 0.5 for all classes.
        ap: AP at IoU thresholds from 0.5 to 0.95 for all classes.
        mp: Mean precision of all classes.
        mr: Mean recall of all classes.
        map50: Mean AP at IoU threshold of 0.5 for all classes.
        map75: Mean AP at IoU threshold of 0.75 for all classes.
        map: Mean AP at IoU thresholds from 0.5 to 0.95 for all classes.
        mean_results: Mean of results, returns mp, mr, map50, map.
        class_result: Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps: mAP of each class.
        fitness: Model fitness as a weighted combination of metrics.
        update: Update metric attributes with new evaluation results.
        curves: Provides a list of curves for accessing specific metrics like precision, recall, F1, etc.
        curves_results: Provide a list of results for accessing specific metrics like precision, recall, F1, etc.
