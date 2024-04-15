# output_file = "/home/m294/Repo/era5/Germany_era5_sd_stl1_p4.nc"


# if __name__ == "__main__":

#     import cdsapi

#     c = cdsapi.Client()

#     c.retrieve(
#         "reanalysis-era5-land",
#         {
#             "product_type": "reanalysis",
#             "format": "netcdf",
#             "variable": ["stl1", "sd"],
#             "year": [str(year) for year in range(2019, 2023)],
#             "month": [str(month).zfill(2) for month in range(1, 13)],
#             "day": [str(day).zfill(2) for day in range(1, 32)],
#             "time": [
#                 "00:00",
#                 "12:00",
#             ],
#             "area": [57.0, 3.0, 43.0, 17.0],
#         },
#         output_file,
#     )
