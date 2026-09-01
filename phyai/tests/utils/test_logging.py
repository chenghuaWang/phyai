"""Unit tests for phyai.utils.logging.

The module mutates process-global logging state (the LogRecord factory, the
root handler), so every test restores what it touched.
"""

from __future__ import annotations

import logging

import pytest

from phyai.utils import logging as plog
from phyai.utils.logging import (
    as_phyai_logger,
    get_logger,
    install_rank_label,
    rank_label,
    reset_rank_label_cache,
)


@pytest.fixture(autouse=True)
def _restore_logging_globals(monkeypatch):
    # Clear the launcher env so the "no label" assertions do not depend on how
    # the suite itself was launched; tests that want it set it explicitly.
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    factory = logging.getLogRecordFactory()
    installed = plog._rank_factory_installed
    reset_rank_label_cache()
    yield
    logging.setLogRecordFactory(factory)
    plog._rank_factory_installed = installed
    reset_rank_label_cache()
    plog._log_once.cache_clear()


@pytest.fixture
def as_rank(monkeypatch):
    """Pretend this process is rank ``r`` of ``w``."""

    def _apply(r: int, w: int = 4):
        monkeypatch.setattr(plog.dist, "is_available", lambda: True)
        monkeypatch.setattr(plog.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(plog.dist, "get_rank", lambda: r)
        monkeypatch.setattr(plog.dist, "get_world_size", lambda: w)
        reset_rank_label_cache()

    return _apply


# --------------------------------------------------------------------------- #
# get_logger                                                                  #
# --------------------------------------------------------------------------- #

EXTRA_METHODS = (
    "log_rank0",
    "debug_rank0",
    "info_rank0",
    "warning_rank0",
    "error_rank0",
    "debug_once",
    "info_once",
    "warning_once",
)


def test_get_logger_attaches_every_extra_method():
    logger = get_logger("phyai.test.attach")
    for name in EXTRA_METHODS:
        assert callable(getattr(logger, name)), name


def test_get_logger_is_idempotent_and_returns_the_singleton():
    first = get_logger("phyai.test.singleton")
    second = get_logger("phyai.test.singleton")
    assert first is second
    assert first is logging.getLogger("phyai.test.singleton")


def test_patching_reaches_a_reference_captured_earlier():
    """Instance patching works on the shared singleton, so an already-held
    reference gains the methods too."""
    captured = logging.getLogger("phyai.test.captured")
    assert not hasattr(captured, "info_rank0")
    get_logger("phyai.test.captured")
    assert hasattr(captured, "info_rank0")


def test_as_phyai_logger_adopts_a_plain_logger():
    plain = logging.getLogger("phyai.test.adopt")
    adopted = as_phyai_logger(plain)
    assert adopted is plain
    assert callable(adopted.info_rank0)


# --------------------------------------------------------------------------- #
# rank_label                                                                  #
# --------------------------------------------------------------------------- #


def test_label_is_empty_without_distributed():
    """Single-process runs must not carry rank noise on every line."""
    assert rank_label() == ""


def test_launcher_env_labels_lines_logged_before_the_group_exists(monkeypatch):
    """Under torchrun, early startup logs precede distributed initialization."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    reset_rank_label_cache()
    assert rank_label() == "[rank 3/4] "


def test_launcher_env_with_world_size_one_stays_quiet(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    reset_rank_label_cache()
    assert rank_label() == ""


def test_env_fallback_is_not_cached_and_the_group_wins(as_rank, monkeypatch):
    """The env is a stand-in; once the real group exists it is authoritative."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    reset_rank_label_cache()
    assert rank_label() == "[rank 3/4] "
    as_rank(1, 2)  # a real group reporting something different
    assert rank_label() == "[rank 1/2] "


def test_empty_label_is_not_cached(as_rank):
    assert rank_label() == ""
    as_rank(2, 8)
    # reset_rank_label_cache inside as_rank proves nothing on its own, so check
    # the real requirement: an early empty label must not stick.
    assert rank_label() == "[rank 2/8] "


def test_label_under_distributed(as_rank):
    as_rank(3, 4)
    assert rank_label() == "[rank 3/4] "


def test_label_is_cached_once_distributed(as_rank, monkeypatch):
    as_rank(3, 4)
    assert rank_label() == "[rank 3/4] "

    def _boom():  # pragma: no cover - must not be reached
        raise AssertionError("rank should be cached after the first resolve")

    monkeypatch.setattr(plog.dist, "get_rank", _boom)
    assert rank_label() == "[rank 3/4] "
    reset_rank_label_cache()
    with pytest.raises(AssertionError):
        rank_label()


# --------------------------------------------------------------------------- #
# install_rank_label                                                          #
# --------------------------------------------------------------------------- #


def test_install_is_idempotent():
    plog._rank_factory_installed = False
    assert install_rank_label() is True
    assert install_rank_label() is False


def test_install_chains_the_previous_factory():
    """A factory installed by the embedding application must keep working."""
    previous = logging.getLogRecordFactory()

    def custom(*args, **kwargs):
        record = previous(*args, **kwargs)
        record.custom_marker = "kept"
        return record

    logging.setLogRecordFactory(custom)
    plog._rank_factory_installed = False
    install_rank_label()

    record = logging.getLogRecordFactory()("n", logging.INFO, "p", 1, "msg", None, None)
    assert record.custom_marker == "kept"
    assert hasattr(record, "rank")


def test_every_record_gets_a_rank_attribute(caplog):
    """Including records from loggers that never went through get_logger."""
    plog._rank_factory_installed = False
    install_rank_label()
    with caplog.at_level(logging.INFO):
        logging.getLogger("third.party.untouched").info("hello")
    assert caplog.records[0].rank == ""


def test_default_format_renders_the_label(as_rank):
    plog._rank_factory_installed = False
    install_rank_label()
    as_rank(1, 2)
    record = logging.getLogRecordFactory()(
        "phyai.x", logging.INFO, "p", 1, "hello", None, None
    )
    rendered = logging.Formatter(plog.DEFAULT_LOG_FORMAT).format(record)
    assert "[rank 1/2] phyai.x: hello" in rendered


# --------------------------------------------------------------------------- #
# rank gating                                                                 #
# --------------------------------------------------------------------------- #


def test_rank0_methods_emit_on_rank_zero(as_rank, caplog):
    as_rank(0, 4)
    logger = get_logger("phyai.test.gate0")
    with caplog.at_level(logging.DEBUG):
        logger.info_rank0("visible")
    assert [r.getMessage() for r in caplog.records] == ["visible"]


def test_rank0_methods_are_silent_off_rank_zero(as_rank, caplog):
    as_rank(1, 4)
    logger = get_logger("phyai.test.gate1")
    with caplog.at_level(logging.DEBUG):
        logger.debug_rank0("no")
        logger.info_rank0("no")
        logger.warning_rank0("no")
        logger.error_rank0("no")
        logger.log_rank0(logging.ERROR, "no")
    assert caplog.records == []


def test_explicit_rank_selects_a_different_rank(as_rank, caplog):
    as_rank(2, 4)
    logger = get_logger("phyai.test.gate2")
    with caplog.at_level(logging.INFO):
        logger.info_rank0("on two", rank=2)
        logger.info_rank0("not on zero", rank=0)
    assert [r.getMessage() for r in caplog.records] == ["on two"]


def test_rank0_methods_emit_without_a_process_group(caplog):
    logger = get_logger("phyai.test.nogroup")
    with caplog.at_level(logging.INFO):
        logger.info_rank0("single process still logs")
    assert len(caplog.records) == 1


def test_levels_are_correct(caplog):
    logger = get_logger("phyai.test.levels")
    with caplog.at_level(logging.DEBUG):
        logger.debug_rank0("d")
        logger.info_rank0("i")
        logger.warning_rank0("w")
        logger.error_rank0("e")
        logger.log_rank0(logging.CRITICAL, "c")
    assert [r.levelno for r in caplog.records] == [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]


def test_lazy_percent_formatting_survives(caplog):
    logger = get_logger("phyai.test.fmt")
    with caplog.at_level(logging.INFO):
        logger.info_rank0("x=%d y=%s", 3, "a")
    assert caplog.records[0].getMessage() == "x=3 y=a"


# --------------------------------------------------------------------------- #
# *_once                                                                      #
# --------------------------------------------------------------------------- #


def test_once_drops_repeats_but_not_new_arguments(caplog):
    plog._log_once.cache_clear()
    logger = get_logger("phyai.test.once")
    with caplog.at_level(logging.WARNING):
        logger.warning_once("falling back to %s", "sdpa")
        logger.warning_once("falling back to %s", "sdpa")
        logger.warning_once("falling back to %s", "torch")
    assert [r.getMessage() for r in caplog.records] == [
        "falling back to sdpa",
        "falling back to torch",
    ]


def test_once_variants_use_their_level(caplog):
    plog._log_once.cache_clear()
    logger = get_logger("phyai.test.once.levels")
    with caplog.at_level(logging.DEBUG):
        logger.debug_once("d")
        logger.info_once("i")
        logger.warning_once("w")
    assert [r.levelno for r in caplog.records] == [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
    ]


def test_once_is_per_logger(caplog):
    plog._log_once.cache_clear()
    a = get_logger("phyai.test.once.a")
    b = get_logger("phyai.test.once.b")
    with caplog.at_level(logging.WARNING):
        a.warning_once("same text")
        b.warning_once("same text")
    assert len(caplog.records) == 2


# --------------------------------------------------------------------------- #
# configure_logging                                                           #
# --------------------------------------------------------------------------- #


def test_configure_logging_installs_the_label_even_when_a_handler_exists():
    """pytest owns the root handler, so this is the branch CI actually takes."""
    plog._rank_factory_installed = False
    plog.configure_logging()
    record = logging.getLogRecordFactory()("n", logging.INFO, "p", 1, "m", None, None)
    assert hasattr(record, "rank")
