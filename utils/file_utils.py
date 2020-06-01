

class cache_property(property):
    """
    descriptor that mimics @property but caches output in member variable

    """

    def __get__(self, obj, objtype=None):
        if not obj:
            return self

        attr = '__cached_' + self.fget.__name__
        cached = getattr(obj, attr, None)
        if not cached:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached

