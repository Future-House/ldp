import sys
from typing import Generic

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from pydantic import BaseModel


# Copied from https://github.com/pydantic/pydantic/issues/9390#issuecomment-2143939391
class PatchGenericPickle:
    """A mixin that allows generic pydantic models to be serialized and deserialized with pickle.

    Notes
    ----
    In general, pickle shouldn't be encouraged as a means of serialization since there are better,
    safer options. In some cases e.g. Streamlit's `@st.cache_data there's no getting around
    needing to use pickle.

    As of Pydantic 2.7, generics don't properly work with pickle. The core issue is the following
    1. For each specialized generic, pydantic creates a new subclass at runtime. This class
       has a `__qualname__` that contains the type var argument e.g. `"MyGeneric[str]"` for a
       `class MyGeneric(BaseModel, Generic[T])`.
    2. Pickle attempts to find a symbol with the value of `__qualname__` in the module where the
       class was defined, which fails since Pydantic defines that class dynamically at runtime.
       Pydantic does attempt to register these dynamic classes but currently only for classes
       defined at the top-level of the interpreter.

    See Also
    --------
    - https://github.com/pydantic/pydantic/issues/9390
    """  # noqa: D416

    @classmethod
    @override
    def __init_subclass__(cls, **kwargs):
        # Note: we're still in __init_subclass__, not yet in __pydantic_init_subclass__
        #  not all model_fields are available at this point.
        super().__init_subclass__(**kwargs)

        if not issubclass(cls, BaseModel):
            raise TypeError(
                "PatchGenericPickle can only be used with subclasses of pydantic.BaseModel"
            )
        if not issubclass(cls, Generic):  # type: ignore[arg-type]
            raise TypeError("PatchGenericPickle can only be used with Generic models")

        qualname = cls.__qualname__
        declaring_module = sys.modules[cls.__module__]
        if qualname not in declaring_module.__dict__:
            # This should work in all cases, but we might need to make this check and update more
            # involved e.g. see pydantic._internal._generics.create_generic_submodel
            declaring_module.__dict__[qualname] = cls
