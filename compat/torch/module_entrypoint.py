"""Executable synthetic-module metadata owned by the compatibility registry."""
import importlib.abc


class ModuleEntrypointLoader(importlib.abc.Loader):
    def __init__(self, fullname, entry_module, entry_name='main'):
        self.fullname = fullname
        self.entry_module = entry_module
        self.entry_name = entry_name
        for qualified in (fullname, entry_module, entry_name):
            if not all(part.isidentifier() for part in qualified.split('.')):
                raise ValueError('entrypoint names must be Python identifiers')

    def create_module(self, spec):
        return None

    def get_code(self, fullname):
        if fullname != self.fullname:
            raise ImportError('loader does not serve ' + fullname)
        source = ('from %s import %s as main\n'
                  'if __name__ == "__main__":\n'
                  '    raise SystemExit(main())\n') % (self.entry_module, self.entry_name)
        return compile(source, '<' + fullname + ' entrypoint>', 'exec')

    def exec_module(self, module):
        exec(self.get_code(module.__name__), module.__dict__)
