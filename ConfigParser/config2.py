from configparser import ConfigParser, ExtendedInterpolation

file = 'config.ini'

config = ConfigParser()
config.read(file)

config2 = ConfigParser(interpolation=ExtendedInterpolation())
config2['settings'] = {
    'debug':'true',
    'secret_key':'abc123',
    'log_path':'/my_app/log',
    'python_version':'3',
    'packages_path':'usr/local'
}

config2['db'] = {
    'db_name':'mongo',
    'db_host':'localhost',
    'db_port':'8889'
}

config2['files'] = {
    'use_cdn':'false',
    'image_path':'/my_app/images',
    'python_path': \
        '${settings:packages_path}/bin/python${settings:python_version}'
}
# config.getfloat()

print(config2['files']['python_path'])
with open("config2.ini", 'w') as f:
    config2.write(f)

