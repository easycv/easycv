class InvalidResource(Exception):
    def __init__(self, resource_name):
        super().__init__("Resource '{}' does not exist.".format(resource_name))


class ErrorDownloadingResource(Exception):
    def __init__(self, reason):
        super().__init__("Error downlaoding the resource. Reason: " + reason)
