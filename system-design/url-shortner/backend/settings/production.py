# project-makefile
from backend.utils import get_ec2_metadata

LOCAL_IPV4 = get_ec2_metadata()
ALLOWED_HOSTS.append(LOCAL_IPV4)
