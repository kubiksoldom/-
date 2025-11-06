.PHONY: test sanity profile

## Run pytest suite quietly
test:
	pytest -q

## Run environment sanity checks without touching live exchanges
sanity:
	python sanity_check.py

## Execute profiling harness if available
profile:
	@if [ -f profiling/run_profiling.py ]; then \
		python profiling/run_profiling.py; \
	elif [ -f run_profiling.py ]; then \
		python run_profiling.py; \
	else \
		echo "profiling runner not found" && exit 1; \
	fi
