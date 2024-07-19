# from kedro.pipeline import Pipeline, node
# from kedro.pipeline.modular_pipeline import pipeline

# from .nodes import forecast_prediction


# def create_pipeline(**kwargs) -> Pipeline:
#     alert_pipeline = pipeline(
#         [
#             node(
#                 forecast_prediction,
#                 inputs=[
#                     "COVID_cases_processing_forecast",
#                     "params:size_PW_daily",
#                     "params:size_FW_weekly",
#                     "params:size_FW_daily",
#                     "params:time_dim_covid_forecast",
#                     "params:size_FW_daily_alert",
#                     "params:age_strat_true",
#                 ],
#                 outputs="Forecast_COVID_cases",
#                 name="Forecast_COVID_cases_node",
#             ),
#             node(
#                 forecast_prediction,
#                 inputs=[
#                     "COVID_deaths_processing_forecast",
#                     "params:size_PW_daily",
#                     "params:size_FW_weekly",
#                     "params:size_FW_daily",
#                     "params:time_dim_covid_forecast",
#                     "params:size_FW_daily_alert",
#                     "params:age_strat_true",
#                 ],
#                 outputs="Forecast_COVID_deaths",
#                 name="Forecast_COVID_deaths_node",
#             ),
#             node(
#                 forecast_prediction,
#                 inputs=[
#                     "COVID_hosp_processing_forecast",
#                     "params:size_PW_daily",
#                     "params:size_FW_weekly",
#                     "params:size_FW_daily",
#                     "params:time_dim_covid_forecast",
#                     "params:size_FW_daily_alert",
#                     "params:age_strat_true",
#                 ],
#                 outputs="Forecast_COVID_hosp",
#                 name="Forecast_COVID_hosp_node",
#             ),
#             node(
#                 forecast_prediction,
#                 inputs=[
#                     "Influenza_cases_processing_forecast",
#                     "params:size_PW_weekly",
#                     "params:size_FW_weekly",
#                     "params:size_FW_daily",
#                     "params:time_dim_influenza_forecast",
#                     "params:size_FW_daily_alert",
#                     "params:age_strat_false",
#                 ],
#                 outputs="Forecast_Influenza_cases",
#                 name="Forecast_Influenza_cases_node",
#             ),
#             node(
#                 forecast_prediction,
#                 inputs=[
#                     "RSV_cases_processing_forecast",
#                     "params:size_PW_weekly",
#                     "params:size_FW_weekly",
#                     "params:size_FW_daily",
#                     "params:time_dim_rsv_forecast",
#                     "params:size_FW_daily_alert",
#                     "params:age_strat_false",
#                 ],
#                 outputs="Forecast_RSV_cases",
#                 name="Forecast_RSV_cases_forecast_node",
#             ),
#             # node(
#             #     forecast_prediction,
#             #     inputs=[
#             #         "SARI_hospit_processing_forecast",
#             #         "params:size_PW_weekly",
#             #         "params:size_FW_weekly",
#             #         "params:size_FW_daily",
#             #         "params:time_dim_SARI_forecast",
#             #         "params:size_FW_daily_alert",
#             #         "params:age_strat_false",
#             #     ],
#             #     outputs="Forecast_SARI_hospit",
#             #     name="Forecast_SARI_hospit_node",
#             # ),
#         ]
#     )
#     return alert_pipeline
