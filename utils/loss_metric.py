from keras import backend as K
import tensorflow as tf

smooth=1

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))
def dice_loss(y_true,y_pred):
  return 1-dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    return - iou(y_true, y_pred)

def precision(y_true,y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    #false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos+0.01)/(true_pos+false_pos+0.01)
def recall(y_true,y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    #false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos+0.1)/(true_pos+false_neg+0.1)

def f1_score(y_true,y_pred):
  return 2*recall(y_true,y_pred)*precision(y_true,y_pred)/(recall(y_true,y_pred)+precision(y_true,y_pred))

