.PHONY: run-qdrant

run:
	echo "Running stack memory agent ..."
	docker-compose --env-file .env -p memory up -d

stop:
	echo "Stopping memory agent ..."
	docker-compose --env-file .env -p memory down

restart: stop run
	echo "All services are restarted"