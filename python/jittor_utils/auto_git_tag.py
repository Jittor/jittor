import subprocess as sp
import os

fdir = os.path.dirname(__file__)
logs = sp.getoutput(f"cd {fdir} && git log -p -- ../jittor/__init__.py ")
# print(logs)

lines = logs.splitlines()

prev_commit = -1
for i in range(len(lines)):
    line = lines[i]
    if line.startswith("+__version__"):
        version = line.split('\'')[1]
        commit = None
        date = None
        msg = []
        for j in range(i,prev_commit,-1):
            if lines[j].startswith("Date:"):
                msg.append(lines[j+2])
        for j in range(i,prev_commit,-1):
            if lines[j].startswith("commit "):
                commit = lines[j].split()[1]
                prev_commit = j + 3
                date = lines[j+2]
                break
        assert commit, version
        print(version, commit)
        msg = msg[::-1]
        cnt = len(msg)
        msg = "\n".join(msg)
        msg = f"Version {version}\n"+date+f"\nTotal {cnt} commits:\n"+msg
        print(msg)
        cmd = f"git tag {version} {commit} -m \"{msg}\""
        print(cmd)
        ret = sp.getoutput(f"cd {fdir} && {cmd}")
        print(ret)
        ret = sp.getoutput(f"cd {fdir} && bash ./github_release.sh {version} \"version {version}\"""")
        print(ret)
        # break