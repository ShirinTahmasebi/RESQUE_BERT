# For creating constant variables (public static final variables)
# https://stackoverflow.com/a/2688086

def constant(f):
    def fset(self, value):
        raise TypeError

    def fget(self):
        return f()
    return property(fget, fset)
