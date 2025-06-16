import pytest
from simdlib import (sum_list, multiply_list, min_list, max_list, any_list, all_list,
                     add_each_list, subtract_each_list, multiply_each_list, square_each_list,
                     subtract_and_square_each_list
                     )

def test_sum():
    assert sum_list([1, 2, 3]) == 6
    assert sum_list([-1, -2, -3]) == -6
    with pytest.raises(SystemError):
        sum_list([]) == 0
    with pytest.raises(SystemError):
        sum_list(["hello", "world"])

def test_multiply():
    assert multiply_list([0, 1, 2, 3]) == 0
    assert multiply_list([1, 2, 3]) == 6
    assert multiply_list([-1, -2, -3]) == -6
    with pytest.raises(SystemError):
        multiply_list([]) == 1
    with pytest.raises(SystemError):
        multiply_list(["hello", "world"])

def test_min():
    assert min_list([0, 1, 2]) == 0
    assert min_list([-1, -2, -3]) == -3
    assert min_list([1, 1, 1]) == 1
    with pytest.raises(SystemError):
        min_list([]) 
    with pytest.raises(SystemError):
        min_list(["hello", "world"])

def test_max():
    assert max_list([1, 2, 3]) == 3
    assert max_list([-1, -2, -3]) == -1
    assert max_list([0, 0, 0]) == 0
    with pytest.raises(SystemError):
        max_list([])
    with pytest.raises(SystemError):
        max_list(["hello", "world"])

def test_any():
    assert any_list([0, 0, 1]) == True
    assert any_list([0, 0, 0]) == False
    with pytest.raises(SystemError):
        any_list([])
    with pytest.raises(SystemError):
        any_list(["hello", "world"])
    
def test_all():
    assert all_list([1, 1, 0]) == False
    assert all_list([1, 1, 1]) == True
    with pytest.raises(SystemError):
        all_list([])
    with pytest.raises(SystemError):
        all_list(["hello", "world"])

def test_add_each():
    assert add_each_list([1, 2, 3], 1) == [2, 3, 4]
    assert add_each_list([0, 0, 0], 0) == [0, 0, 0]
    with pytest.raises(SystemError):
        add_each_list([], 1)

def test_subtract_each():
    assert subtract_each_list([1.0, 2.0, 3.0], 1.0) == [0.0, 1.0, 2.0]
    assert subtract_each_list([0.0, 0.0, 0.0], 0) == [0.0, 0.0, 0.0]
    with pytest.raises(SystemError):
        subtract_each_list([], 0)

def test_multiply_each():
    assert multiply_each_list([1, 2, 3], 2) == [2, 4, 6]
    assert multiply_each_list([1, 2, 3], 1) == [1, 2, 3]
    with pytest.raises(SystemError):
        multiply_each_list([], 0)

def test_square_each():
    assert square_each_list([1.0, 2.0, 3.0]) == [1.0, 4.0, 9.0]
    assert square_each_list([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
    with pytest.raises(SystemError):
        square_each_list([])

def test_subtract_and_square_each():
    assert subtract_and_square_each_list([1.0, 2.0, 3.0], 1) == [0.0, 1.0, 4.0]
    assert subtract_and_square_each_list([1.0, 2.0, 3.0], 0) == [1.0, 4.0, 9.0]
    with pytest.raises(SystemError):
        subtract_and_square_each_list([], 0)