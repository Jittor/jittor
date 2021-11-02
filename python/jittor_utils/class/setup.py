import sys
import os
command = sys.argv[1]
if (command == 'ssh'):
    port = sys.argv[2]
    data = open("/etc/ssh/sshd_config", "r").readlines()
    data[12] = 'Port ' + port + '\nPermitRootLogin yes\n' 
    f = open("/etc/ssh/sshd_config", "w")
    f.writelines(data)
    f.close()
    os.system("service ssh restart")
elif (command == 'passwd'):
    passwd = sys.argv[2]
    os.system("echo root:"+passwd+" | chpasswd")
else:
    print('command error')
