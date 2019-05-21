# Yolo
Use yolo for prediction.

  There are mainly three parts of the code.
  1. Yolo weights extractor: transform weights file (tiny-v2: https://pjreddie.com/media/files/yolov2-tiny.weights) from DarkNet to tf;
  
    >> Use np.fromfile() read the .weights file; useful weights data start from 6th;
    
    >> Structure of .weights file: for batchnorm layer, data are in sequence of beta, gamma, mean, variance for batchnormalization, and then weights for kernel; for layer without batchnorm, they are bias and then weights;
    
    >> Different from case in tf, weights for kernel from DarkNet are in structure of [outfilter, infilter, size, size]. So after reshape, a transpose is needed. 
    
  2. Build net according to cfg file from DarkNet (tiny-v2: https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg);
  3. Postprocessing: interpret output from nets to boxes.
  
  Tips:
  1. Inputs need to be normalized to 0-1;
  2. For max_pool layer with size 2 and stride 1, padding is needed to keep img size invariant (tf.nn.max_pool(-,-,-, padding = 'SAME'));
  3. For conv layer with size 1, stride 1 and pad 1, use argument tf.nn.conv2d(-,-,-,padding = 'SAME'), which means there is no padding in fact.
