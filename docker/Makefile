.PHONY: run stop restart dev stop-dev

PLATFORM ?= cpu

ifeq ($(filter $(PLATFORM),cpu gpu dev),)
$(error PLATFORM must be 'cpu' or 'gpu' or 'dev' (got '$(PLATFORM)'))
endif

dev:
	@echo "Running stack memory agent ... "
	@docker-compose -p memory-agent --profile dev --profiles $(PLATFORM) up -d

stop-dev:
	@echo "Stopping stack memory agent ... "
	@docker-compose -p memory-agent --profile dev down

run:
	@echo "Running stack memory agent ... (platform=$(PLATFORM))"
	@docker-compose -p memory-agent --profile all --profiles $(PLATFORM) pull
	@docker-compose -p memory-agent --profile all --profiles $(PLATFORM) up -d

stop:
	@echo "Stopping memory agent ... (platform=$(PLATFORM))"
	@docker-compose -p memory-agent --profile all --profiles $(PLATFORM) down

restart: stop run
	@echo "All services are restarted (platform=$(PLATFORM))"