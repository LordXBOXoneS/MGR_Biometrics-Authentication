# Generated by Django 5.1.4 on 2025-01-31 15:02

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('auth_app', '0002_userprofile_delete_note'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userprofile',
            name='phone',
        ),
    ]
