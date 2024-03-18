export_conda:
	conda env export > "conda/environment_$(shell whoami)_$(shell date +%Y-%m-%d_%H:%M).yml"
