from easycv import List
from easycv.transforms import GrayScale, Blur

testlist = List.random(2)
lazy_test_list = List.random(2, lazy=True)


def test_random():
    test_list = testlist.copy()
    assert len(test_list) == 2


def test_start_shutdown():
    List.start()
    List.shutdown()


def test_index():
    test_list = testlist.copy()
    assert len(test_list[:1]) == 1


def test_apply():
    test_list = testlist.copy()
    t = test_list.apply(GrayScale())
    assert len(t) == 2
    t2 = t.copy()
    t2.apply(Blur(), in_place=True)
    assert id(t) != id(t2)


def test_compute():
    test_list = lazy_test_list.copy()
    assert not test_list[0].loaded
    test_list.apply(Blur(), in_place=True)
    assert not test_list[0].loaded
    assert test_list[0].pending.num_transforms() == 1
    test_list.compute(in_place=True)
    assert test_list[0].pending.num_transforms() == 0
    assert len(test_list) == 2


def parallel():
    test_list = testlist.copy()
    t = test_list.apply(GrayScale(), parallel=True)
    assert len(t) == 2
    t2 = t.copy()
    t2.apply(Blur(), in_place=True, parallel=True)
    assert id(t) != id(t2)

    test_list = lazy_test_list.copy()
    assert not test_list[0].loaded
    test_list.apply(Blur(), in_place=True, parallel=True)
    assert not test_list[0].loaded
    assert test_list[0].pending.num_transforms() == 1
    test_list.compute(in_place=True, parallel=True)
    assert test_list[0].pending.num_transforms() == 0
    assert len(test_list) == 2
    List.shutdown()
