from __future__ import annotations


class NumbersDict:
    """
    It's mostly like dict[str,float|int] & some optional broadcast default value.
    It implements the standard math bin ops in a straight-forward way.
    """

    def __init__(self, auto_convert=None, numbers_dict=None, broadcast_value=None):
        """
        :param dict|NumbersDict|T auto_convert: first argument, so that we can automatically convert/copy
        :param dict numbers_dict:
        :param T broadcast_value:
        """
        if auto_convert is not None:
            assert broadcast_value is None
            assert numbers_dict is None
            if isinstance(auto_convert, dict):
                numbers_dict = auto_convert
            elif isinstance(auto_convert, NumbersDict):
                numbers_dict = auto_convert.dict
                broadcast_value = auto_convert.value
            else:
                broadcast_value = auto_convert
        if numbers_dict is None:
            numbers_dict = {}
        else:
            numbers_dict = dict(numbers_dict)  # force copy

        self.dict = numbers_dict
        self.value = broadcast_value
        self.max = self._max_error

    def copy(self):
        """
        :rtype: NumbersDict
        """
        return NumbersDict(self)

    @classmethod
    def constant_like(cls, const_number, numbers_dict):
        """
        :param int|float|object const_number:
        :param NumbersDict numbers_dict:
        :return: NumbersDict with same keys as numbers_dict
        :rtype: NumbersDict
        """
        return NumbersDict(
            broadcast_value=const_number if (numbers_dict.value is not None) else None,
            numbers_dict={k: const_number for k in numbers_dict.dict.keys()},
        )

    def copy_like(self, numbers_dict):
        """
        :param NumbersDict numbers_dict:
        :return: copy of self with same keys as numbers_dict as far as we have them
        :rtype: NumbersDict
        """
        if self.value is not None:
            return NumbersDict(
                broadcast_value=self.value if (numbers_dict.value is not None) else None,
                numbers_dict={k: self[k] for k in numbers_dict.dict.keys()},
            )
        else:
            return NumbersDict(
                broadcast_value=None, numbers_dict={k: self[k] for k in numbers_dict.dict.keys() if k in self.dict}
            )

    @property
    def keys_set(self):
        """
        :rtype: set[str]
        """
        return set(self.dict.keys())

    def __getitem__(self, key):
        if self.value is not None:
            return self.dict.get(key, self.value)
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __delitem__(self, key):
        del self.dict[key]

    def get(self, key, default=None):
        """
        :param str key:
        :param T default:
        :rtype: object|T
        """
        # Keep consistent with self.__get_item__. If self.value is set, this will always be the default value.
        return self.dict.get(key, self.value if self.value is not None else default)

    def pop(self, key, *args):
        """
        :param str key:
        :param T args: default, or not
        :rtype: object|T
        """
        return self.dict.pop(key, *args)

    def __iter__(self):
        # This can potentially cause confusion. So enforce explicitness.
        # For a dict, we would return the dict keys here.
        # Also, max(self) would result in a call to self.__iter__(),
        # which would only make sense for our values, not the dict keys.
        raise Exception("%s.__iter__ is undefined" % self.__class__.__name__)

    def keys(self):
        """
        :rtype: set[str]
        """
        return self.dict.keys()

    def values(self):
        """
        :rtype: list[object]
        """
        return list(self.dict.values()) + ([self.value] if self.value is not None else [])

    def items(self):
        """
        :return: dict items. this excludes self.value
        :rtype: str[(str,object)]
        """
        return self.dict.items()

    def has_values(self):
        """
        :rtype: bool
        """
        return bool(self.dict) or self.value is not None

    def unary_op(self, op):
        """
        :param (T)->T2 op:
        :return: new NumbersDict, where ``op`` is applied on all values
        :rtype: NumbersDict
        """
        res = NumbersDict()
        if self.value is not None:
            res.value = op(self.value)
        for k, v in self.dict.items():
            res.dict[k] = op(v)
        return res

    @classmethod
    def bin_op_scalar_optional(cls, self, other, zero, op):
        """
        :param T self:
        :param T other:
        :param T zero:
        :param (T,T)->T op:
        :rtype: T
        """
        if self is None and other is None:
            return None
        if self is None:
            self = zero
        if other is None:
            other = zero
        return op(self, other)

    @classmethod
    def bin_op(cls, self, other, op, zero, result=None):
        """
        :param NumbersDict|int|float|T self:
        :param NumbersDict|int|float|T other:
        :param (T,T)->T op:
        :param T zero:
        :param NumbersDict|None result:
        :rtype: NumbersDict
        """
        if not isinstance(self, NumbersDict):
            if isinstance(other, NumbersDict):
                self = NumbersDict.constant_like(self, numbers_dict=other)
            else:
                self = NumbersDict(self)
        if not isinstance(other, NumbersDict):
            other = NumbersDict.constant_like(other, numbers_dict=self)
        if result is None:
            result = NumbersDict()
        assert isinstance(result, NumbersDict)
        for k in self.keys_set | other.keys_set:
            result[k] = cls.bin_op_scalar_optional(self.get(k, None), other.get(k, None), zero=zero, op=op)
        result.value = cls.bin_op_scalar_optional(self.value, other.value, zero=zero, op=op)
        return result

    def __add__(self, other):
        return self.bin_op(self, other, op=lambda a, b: a + b, zero=0)

    __radd__ = __add__

    def __iadd__(self, other):
        return self.bin_op(self, other, op=lambda a, b: a + b, zero=0, result=self)

    def __sub__(self, other):
        return self.bin_op(self, other, op=lambda a, b: a - b, zero=0)

    def __rsub__(self, other):
        return self.bin_op(self, other, op=lambda a, b: b - a, zero=0)

    def __isub__(self, other):
        return self.bin_op(self, other, op=lambda a, b: a - b, zero=0, result=self)

    def __mul__(self, other):
        return self.bin_op(self, other, op=lambda a, b: a * b, zero=1)

    __rmul__ = __mul__

    def __imul__(self, other):
        return self.bin_op(self, other, op=lambda a, b: a * b, zero=1, result=self)

    def __div__(self, other):
        return self.bin_op(self, other, op=lambda a, b: a / b, zero=1)

    __rdiv__ = __div__
    __truediv__ = __div__

    def __idiv__(self, other):
        return self.bin_op(self, other, op=lambda a, b: a / b, zero=1, result=self)

    __itruediv__ = __idiv__

    def __floordiv__(self, other):
        return self.bin_op(self, other, op=lambda a, b: a // b, zero=1)

    def __ifloordiv__(self, other):
        return self.bin_op(self, other, op=lambda a, b: a // b, zero=1, result=self)

    def __neg__(self):
        return self.unary_op(op=lambda a: -a)

    def __bool__(self):
        return any(self.values())

    __nonzero__ = __bool__  # Python 2

    def elem_eq(self, other, result_with_default=True):
        """
        Element-wise equality check with other.
        Note about broadcast default value: Consider some key which is neither in self nor in other.
          This means that self[key] == self.default, other[key] == other.default.
          Thus, in case that self.default != other.default, we get res.default == False.
          Then, all(res.values()) == False, even when all other values are True.
          This is sometimes not what we want.
          You can control the behavior via result_with_default.

        :param NumbersDict|T other:
        :param bool result_with_default:
        :rtype: NumbersDict
        """

        def op(a, b):
            """
            :param a:
            :param b:
            :rtype: bool|None
            """
            if a is None:
                return None
            if b is None:
                return None
            return a == b

        res = self.bin_op(self, other, op=op, zero=None)
        if not result_with_default:
            res.value = None
        return res

    def __eq__(self, other):
        """
        :param NumbersDict|T other:
        :return: whether self == other elemwise. see self.elem_eq
        :rtype: bool
        """
        return all(self.elem_eq(other).values())

    def __ne__(self, other):
        """
        :param NumbersDict|T other:
        :return: not (self == other)
        :rtype: bool
        """
        return not (self == other)

    def __cmp__(self, other):
        # There is no good straight-forward implementation
        # and it would just confuse.
        raise Exception("%s.__cmp__ is undefined" % self.__class__.__name__)

    def any_compare(self, other, cmp):
        """
        :param NumbersDict other:
        :param ((object,object)->True) cmp:
        :rtype: True
        """
        for key in self.keys():
            if key in other.keys():
                if cmp(self[key], other[key]):
                    return True
            elif other.value is not None:
                if cmp(self[key], other.value):
                    return True
        if self.value is not None and other.value is not None:
            if cmp(self.value, other.value):
                return True
        return False

    @staticmethod
    def _max(*args):
        args = [a for a in args if a is not None]
        if not args:
            return None
        if len(args) == 1:
            return args[0]
        return max(*args)

    @staticmethod
    def _min(*args):
        args = [a for a in args if a is not None]
        if not args:
            return None
        if len(args) == 1:
            return args[0]
        return min(*args)

    @classmethod
    def max(cls, items):
        """
        Element-wise maximum for item in items.
        :param list[NumbersDict|int|float] items:
        :rtype: NumbersDict
        """
        assert items
        if len(items) == 1:
            return NumbersDict(items[0])
        if len(items) == 2:
            return cls.bin_op(items[0], items[1], op=cls._max, zero=None)
        return cls.max([items[0], cls.max(items[1:])])

    @classmethod
    def min(cls, items):
        """
        Element-wise minimum for item in items.
        :param list[NumbersDict|int|float] items:
        :rtype: NumbersDict
        """
        assert items
        if len(items) == 1:
            return NumbersDict(items[0])
        if len(items) == 2:
            return cls.bin_op(items[0], items[1], op=cls._min, zero=None)
        return cls.min([items[0], cls.min(items[1:])])

    @staticmethod
    def _max_error():
        # Will replace self.max for each instance. To be sure that we don't confuse it with self.max_value.
        raise Exception("Use max_value instead.")

    def max_value(self):
        """
        Maximum of our values.
        """
        return max(self.values())

    def min_value(self):
        """
        Minimum of our values.
        """
        return min(self.values())

    def __repr__(self):
        if self.value is None and not self.dict:
            return "%s()" % self.__class__.__name__
        if self.value is None and self.dict:
            return "%s(%r)" % (self.__class__.__name__, self.dict)
        if not self.dict and self.value is not None:
            return "%s(%r)" % (self.__class__.__name__, self.value)
        return "%s(numbers_dict=%r, broadcast_value=%r)" % (self.__class__.__name__, self.dict, self.value)
