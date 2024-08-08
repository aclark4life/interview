"""
Django settings for backend project.

Generated by 'django-admin startproject' using Django 5.1.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.1/ref/settings/
"""

from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-2)39qg2t@xka)ggqpup(%=$y^(kmt^k$%#z^=-sy1+lk5===m8"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "url_shortener",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "backend.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL = "static/"

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
import os

STATICFILES_DIRS = []
THEMES = [
    ("light", "Light Theme"),
    ("dark", "Dark Theme"),
]
INSTALLED_APPS.append("home")
SEARCH_MODELS = [
    # 'app_name.Article',
    # 'app_name.BlogPost',
    # 'app_name.Comment',
    # Add other models as needed
]
# project-makefile
ALLOWED_HOSTS = ["*"]
import dj_database_url  # noqa

DATABASE_URL = os.environ.get("DATABASE_URL", "postgres://:@:/project-makefile")
DATABASES["default"] = dj_database_url.parse(DATABASE_URL)
INSTALLED_APPS.append("webpack_boilerplate")
INSTALLED_APPS.append("rest_framework")
INSTALLED_APPS.append("rest_framework.authtoken")
INSTALLED_APPS.append("allauth")
INSTALLED_APPS.append("allauth.account")
INSTALLED_APPS.append("allauth.socialaccount")
INSTALLED_APPS.append("django_extensions")
INSTALLED_APPS.append("crispy_forms")
INSTALLED_APPS.append("crispy_bootstrap5")
INSTALLED_APPS.append("django_recaptcha")
# INSTALLED_APPS = [app for app in INSTALLED_APPS if app != 'django.contrib.admin']
# INSTALLED_APPS.append('backend.apps.CustomAdminConfig')
MIDDLEWARE.append("allauth.account.middleware.AccountMiddleware")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(PROJECT_DIR)
STATICFILES_DIRS.append(os.path.join(BASE_DIR, "frontend/build"))
WEBPACK_LOADER = {
    "MANIFEST_FILE": os.path.join(BASE_DIR, "frontend/build/manifest.json"),
}
REST_FRAMEWORK = {
    # Use Django's standard `django.contrib.auth` permissions,
    # or allow read-only access for unauthenticated users.
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.DjangoModelPermissionsOrAnonReadOnly"
    ]
}
THEMES = [
    ("light", "Light Theme"),
    ("dark", "Dark Theme"),
]
LOGIN_REDIRECT_URL = "/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
]
SILENCED_SYSTEM_CHECKS = ["django_recaptcha.recaptcha_test_key_error"]
EXPLORER_CONNECTIONS = {"Default": "default"}
EXPLORER_DEFAULT_CONNECTION = "default"
TEMPLATES[0]["DIRS"].append(os.path.join(PROJECT_DIR, "templates"))
INSTALLED_APPS.append("siteuser")
AUTH_USER_MODEL = "siteuser.User"
