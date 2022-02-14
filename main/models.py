from distutils.command.upload import upload
from django.db import models

# Create your models here.
class DogIdentify(models.Model):
    id = models.AutoField(primary_key=True)
    user_ip = models.CharField(max_length=50)
    dog_pic = models.ImageField(upload_to = 'dog_img')