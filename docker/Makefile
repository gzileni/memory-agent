.PHONY: run-qdrant

run:
	echo "Running stack agent ..."
	docker-compose -p memory up -d

stop:
	echo "Stopping qdrant ..."
	docker-compose down

restart: stop run
	echo "All services are restarted"