# Generated by Django 5.1 on 2024-08-08 23:30

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="URLMapping",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("original_url", models.URLField(max_length=2048)),
                ("shortened_url", models.CharField(max_length=10, unique=True)),
                ("access_count", models.IntegerField(default=0)),
            ],
        ),
    ]