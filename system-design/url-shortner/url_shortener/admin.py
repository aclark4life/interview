# Register your models here.

from django.contrib import admin
from .models import URLMapping


@admin.register(URLMapping)
class URLMappingAdmin(admin.ModelAdmin):
    list_display = ("shortened_url", "original_url", "access_count")
