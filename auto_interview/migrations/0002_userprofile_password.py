# Generated by Django 4.1.5 on 2024-02-25 06:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auto_interview', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='password',
            field=models.CharField(blank=True, max_length=128, null=True),
        ),
    ]
