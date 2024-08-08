# Create your models here.
from django.db import models


class URLMapping(models.Model):
    original_url = models.URLField(max_length=2048)
    shortened_url = models.CharField(max_length=10, unique=True)
    access_count = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.shortened_url} -> {self.original_url}"
