# Generated by Django 4.0.2 on 2022-02-13 02:13

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DogIdentify',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user_ip', models.CharField(max_length=50)),
                ('dog_pic', models.ImageField(upload_to='dog')),
            ],
        ),
    ]
