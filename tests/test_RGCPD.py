#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the RGCPD module.
"""
import pytest

from RGCPD import RGCPD


def test_without_test_object():
    assert False


class TestRgcpd(object):
    @pytest.fixture
    def return_a_test_object(self):
        pass

    def test_RGCPD(self, RGCPD):
        assert False

    def test_with_error(self, RGCPD):
        with pytest.raises(ValueError):
            pass
