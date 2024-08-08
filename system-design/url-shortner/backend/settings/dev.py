from .base import *  # noqa

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-#t%ohiokp+8!7#xh4qzoxuyy=-&sxl*!z-&w%y83h87-jm7p9="

# SECURITY WARNING: define the correct hosts in production!
ALLOWED_HOSTS = ["*"]  # noqa

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"


try:
    from .local import *
except ImportError:
    pass
INSTALLED_APPS.append("debug_toolbar")  # noqa
INSTALLED_APPS.append("explorer")  # noqa
INSTALLED_APPS.append("django.contrib.admindocs")  # noqa
MIDDLEWARE.append("debug_toolbar.middleware.DebugToolbarMiddleware")  # noqa
MIDDLEWARE.append("hijack.middleware.HijackUserMiddleware")  # noqa
INTERNAL_IPS = [
    "127.0.0.1",
]
