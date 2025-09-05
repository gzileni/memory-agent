.PHONY: run stop restart log stop-log

PLATFORM ?= cpu

ifeq ($(filter $(PLATFORM),cpu gpu log),)
$(error PLATFORM must be 'cpu' or 'gpu' or 'log' (got '$(PLATFORM)'))
endif

log:
	@echo "Running log stack ... "
	@docker-compose -p memory-agent --profile log up -d

stop-log:
	@echo "Stopping log stack ... "
	@docker-compose -p memory-agent --profile log down

run:
	@echo "Running stack memory agent ... (platform=$(PLATFORM))"
	@docker-compose -p memory-agent --profile all --profile $(PLATFORM) pull
	@docker-compose -p memory-agent --profile all --profile $(PLATFORM) up -d

stop:
	@echo "Stopping memory agent ... (platform=$(PLATFORM))"
	@docker-compose -p memory-agent --profile all --profile $(PLATFORM) down

restart: stop run
	@echo "All services are restarted (platform=$(PLATFORM))"