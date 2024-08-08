from django.conf import settings
from django.urls import include, path
from django.contrib import admin
from url_shortener.api import api

urlpatterns = []
if settings.DEBUG:
    urlpatterns += [
        path("django/doc/", include("django.contrib.admindocs.urls")),
    ]
urlpatterns += [
    path("accounts/", include("allauth.urls")),
    path("django/", admin.site.urls),
    path("user/", include("siteuser.urls")),
    path("explorer/", include("explorer.urls")),
    path("hijack/", include("hijack.urls")),
    path("search/", include("search.urls")),
    path("api/", api.urls),
    path("", include("home.urls")),
]
if settings.DEBUG:
    from django.conf.urls.static import static
    from django.contrib.staticfiles.urls import staticfiles_urlpatterns

    # Serve static and media files from development server
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    import debug_toolbar

    urlpatterns += [
        path("__debug__/", include(debug_toolbar.urls)),
    ]
