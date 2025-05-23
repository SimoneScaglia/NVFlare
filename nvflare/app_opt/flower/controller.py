# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tie.controller import TieController
from nvflare.app_common.tie.defs import Constant as TieConstant
from nvflare.app_opt.flower.applet import FlowerServerApplet
from nvflare.app_opt.flower.connectors.grpc_server_connector import GrpcServerConnector
from nvflare.fuel.utils.validation_utils import check_positive_number

from .defs import Constant


class FlowerController(TieController):
    def __init__(
        self,
        num_rounds=1,
        database: str = "",
        superlink_ready_timeout: float = 10.0,
        superlink_grace_period: float = 2.0,
        superlink_min_query_interval=10.0,
        monitor_interval: float = 0.5,
        configure_task_name=TieConstant.CONFIG_TASK_NAME,
        configure_task_timeout=TieConstant.CONFIG_TASK_TIMEOUT,
        start_task_name=TieConstant.START_TASK_NAME,
        start_task_timeout=TieConstant.START_TASK_TIMEOUT,
        job_status_check_interval: float = TieConstant.JOB_STATUS_CHECK_INTERVAL,
        max_client_op_interval: float = TieConstant.MAX_CLIENT_OP_INTERVAL,
        progress_timeout: float = TieConstant.WORKFLOW_PROGRESS_TIMEOUT,
        int_client_grpc_options=None,
    ):
        """Constructor of FlowerController

        Args:
            num_rounds: number of rounds. Not used in this version.
            database: database name
            superlink_ready_timeout: how long to wait for the superlink to become ready before starting server app
            superlink_min_query_interval: minimal interval for querying superlink for status
            monitor_interval: how often to check flower run status
            configure_task_name: name of the config task
            configure_task_timeout: max time allowed for config task to complete
            start_task_name: name of the start task
            start_task_timeout: max time allowed for start task to complete
            job_status_check_interval: how often to check job status
            max_client_op_interval: max time allowed for missing client requests
            progress_timeout: max time allowed for missing overall progress
            int_client_grpc_options: internal grpc client options
        """
        TieController.__init__(
            self,
            configure_task_name=configure_task_name,
            configure_task_timeout=configure_task_timeout,
            start_task_name=start_task_name,
            start_task_timeout=start_task_timeout,
            job_status_check_interval=job_status_check_interval,
            max_client_op_interval=max_client_op_interval,
            progress_timeout=progress_timeout,
        )

        check_positive_number("superlink_ready_timeout", superlink_ready_timeout)
        check_positive_number("superlink_grace_period", superlink_grace_period)
        check_positive_number("monitor_interval", monitor_interval)
        check_positive_number("superlink_min_query_interval", superlink_min_query_interval)

        self.num_rounds = num_rounds
        self.database = database
        self.superlink_ready_timeout = superlink_ready_timeout
        self.superlink_grace_period = superlink_grace_period
        self.superlink_min_query_interval = superlink_min_query_interval
        self.int_client_grpc_options = int_client_grpc_options
        self.monitor_interval = monitor_interval

    def get_connector(self, fl_ctx: FLContext):
        return GrpcServerConnector(
            int_client_grpc_options=self.int_client_grpc_options,
            monitor_interval=self.monitor_interval,
        )

    def get_applet(self, fl_ctx: FLContext):
        return FlowerServerApplet(
            database=self.database,
            superlink_ready_timeout=self.superlink_ready_timeout,
            superlink_grace_period=self.superlink_grace_period,
            superlink_min_query_interval=self.superlink_min_query_interval,
        )

    def get_client_config_params(self, fl_ctx: FLContext) -> dict:
        return {
            Constant.CONF_KEY_NUM_ROUNDS: self.num_rounds,
        }

    def get_connector_config_params(self, fl_ctx: FLContext) -> dict:
        return {
            Constant.CONF_KEY_NUM_ROUNDS: self.num_rounds,
        }
