.PHONY: sync_deps
sync_deps:
	uv export --format requirements.txt --output-file requirements.txt