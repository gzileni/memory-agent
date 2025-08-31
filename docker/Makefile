.PHONY: run stop restart

PLATFORM ?= cpu

ifeq ($(filter $(PLATFORM),cpu gpu),)
$(error PLATFORM must be 'cpu' or 'gpu' (got '$(PLATFORM)'))
endif

run:
	@echo "Running stack mcp server ... (platform=$(PLATFORM))"
	@docker-compose -p memory-agent --profile all --profiles $(PLATFORM) pull
	@docker-compose -p memory-agent --profile all --profiles $(PLATFORM) up -d

stop:
	@echo "Stopping mcp server ... (platform=$(PLATFORM))"
	@docker-compose -p memory-agent --profile all --profiles $(PLATFORM) down

restart: stop run
	@echo "All services are restarted (platform=$(PLATFORM))"