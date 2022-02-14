from django.shortcuts import render, redirect
from main.models import DogIdentify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pickle
import wikipedia

# Create your views here.

def home(request):
    if 'userip' in request.session:
      try:
        di = DogIdentify.objects.get(user_ip=request.session['userip'])
        dog_pic = f"media/{di.dog_pic}"
        custom_image_path=[dog_pic]
        model = pickle.load(open('main/model','rb'))
        custom_data=create_data_batches(custom_image_path,test_data=True)
        custom_preds=model.predict(custom_data)
        custom_pred_labels=[get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
        db = custom_pred_labels[0].replace('_', ' ')
        dog_breed = db.title()
        details_dog = wikipedia.summary(f'dog {dog_breed}')
        details_dog.replace("\n"," ")
        return render(request, 'index.html', {'dog_url' : dog_pic, 'ip_addr' : request.session['userip'], 'dog': dog_breed, 'dog_details': details_dog})
      except:
        return render(request, 'index.html', {'dog_url': None, 'ip_addr': 'not available'})
    else:
        return render(request, 'index.html', {'dog_url': None, 'ip_addr': 'not available'})

def identify_breed(request):
    user = get_ip(request)
    print(f"Identify {user}")
    if request.method == 'POST':
        pic = request.FILES['dog_pic']
        try:
            user_get = DogIdentify.objects.get(user_ip=user)
            user_get.dog_pic = pic
            user_get.save()
            request.session['userip'] = user
            return redirect('/')
        except:
            dog_identify = DogIdentify(user_ip=user, dog_pic=pic)
            dog_identify.save()
            request.session['userip'] = user
            return redirect('/')
    else:
        return render(request, 'index.html', {'dog_url': None})

def get_ip(request):
    forwarded_addresses = request.META.get('HTTP_X_FORWARDED_FOR')
    if forwarded_addresses:
        client_addr = forwarded_addresses.split(',')[0]
    else:
        client_addr = request.META.get('REMOTE_ADDR')
    print(client_addr)

    return client_addr

def load_model(model_path):
  """
  Loads the saved model from specified path
  """
  print(f'Loading saved model from {model_path}')
  model=tf.keras.models.load_model(model_path,
                                   custom_objects={'KerasLayer':hub.KerasLayer})
  return model


BATCH_SIZE=32
IMG_SIZE=224


#create a function to turn data into batches
def create_data_batches(x,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):
  """
  creates batches of data out of image(x) and label(y) pairs
  Shuffles the data if its training data but doesn't shuffle if its validation data.
  Also accepts test data as inputs(no labels).
  """
  # if the data is test dataset we probably dont have any labels
  if test_data:
    print('Creating test data batches . . .')
    data=tf.data.Dataset.from_tensor_slices(tf.constant(x))
    data_batch=data.map(process_image).batch(batch_size)
    return data_batch

    #if the data is valid dataset then we dont need to shuffle it
  elif valid_data:
     print('Creating Valid data batches . . .')
     data=tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
     data_batch=data.map(get_image_label).batch(batch_size)
     return data_batch

    
  else:
      print('Creating train data batches . . .')
      #turn filepaths and labels into tensors
      data=tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
      data=data.shuffle(buffer_size=len(x))
      data_batch=data.map(get_image_label).batch(batch_size)
      return data_batch

def process_image(file_path,img_size=IMG_SIZE):
  #read the image file
  image=tf.io.read_file(file_path) #reads and outputs the entire content of given filename

  # Turn the jpeg image into numeric Tensor with 3 color channel(Red,Green,Blue)
  image=tf.image.decode_jpeg(image,channels=3)

  #convert the color channel values from 0-255 to 0-1 values
  image=tf.image.convert_image_dtype(image,dtype=tf.float32)

  #resize  our image to our desired value (244,244)
  image=tf.image.resize(image,size=(img_size,img_size))

  return image

def get_image_label(image_path,label):
  """ takes an image file path and the associated label,
  process the image and returns the tuple (image,label).
  """
  image=process_image(image_path)
  return image,label

def get_pred_label(prediction_probabilities):
  """
  Turn the array of predictionprobabilities into labels
  """
  labels_csv=pd.read_csv('main/labels.csv')
  labels=labels_csv['breed'].to_numpy()
  unique_breeds=np.unique(labels)

  return unique_breeds[np.argmax(prediction_probabilities)]