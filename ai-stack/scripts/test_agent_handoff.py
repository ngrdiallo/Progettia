def test_escalation_blocked():
    router = LocalModelRouter('config.json')
    # Simulate repeated handoff pairs to trigger escalation guard
    router._handoff_pairs = [('A','B'),('B','A'),('A','B'),('B','A')]
    router._escalation_threshold = 2
    try:
        router._enforce_escalation_guard('A','B')
        print("FAIL: Escalation guard did not raise error")
        return False
    except RouterError:
        return True
    except Exception as e:
        print(f"FAIL: Unexpected exception: {e}")
        return False

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import time
from router_core import LocalModelRouter, RouterError


def test_valid_handoff():
    # Simulate a valid handoff scenario
    router = LocalModelRouter('config.json')
    payload = {
        'task': 'summarize',
        'context_min': {'input': 'Test input'},
        'expected_output': 'summary',
        'from_agent': 'agentA',
        'to_agent': 'agentB',
    }
    # Should not raise
    try:
        router._validate_handoff_schema(payload)
    except Exception as e:
        print(f"FAIL: Valid handoff raised: {e}")
        return False
    return True


def test_invalid_schema():
    router = LocalModelRouter('config.json')
    payload = {'task': 't'}  # missing required fields
    try:
        router._validate_handoff_schema(payload)
        print("FAIL: Invalid schema did not raise error")
        return False
    except RouterError:
        return True
    except Exception as e:
        print(f"FAIL: Unexpected exception: {e}")
        return False


def test_budget_exceeded():
    router = LocalModelRouter('config.json')
    router._handoff_count = 5
    router._handoff_budget = 3
    try:
        router._enforce_handoff_budget()
        print("FAIL: Budget exceeded did not raise error")
        return False
    except RouterError:
        return True
    except Exception as e:
        print(f"FAIL: Unexpected exception: {e}")
        return False


def test_timeout_path():
    router = LocalModelRouter('config.json')
    router._handoff_start_time = time.time() - 2
    router._handoff_timeout = 1
    try:
        router._enforce_handoff_timeout()
        print("FAIL: Timeout did not raise error")
        return False
    except RouterError:
        return True
    except Exception as e:
        print(f"FAIL: Unexpected exception: {e}")
        return False


def main():
    results = []
    results.append(test_valid_handoff())
    results.append(test_invalid_schema())
    results.append(test_budget_exceeded())
    results.append(test_timeout_path())
    results.append(test_escalation_blocked())
    if all(results):
        print("AGENT_HANDOFF_TEST_OK")
    else:
        print("FAIL")

if __name__ == "__main__":
    main()


