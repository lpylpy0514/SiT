from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/ymz/newdisk2/for_lpy/data/got10k_lmdb'
    settings.got10k_path = '/home/ymz/newdisk2/for_lpy/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/ymz/newdisk2/for_lpy/data/itb'
    settings.lasot_extension_subset_path_path = '/home/ymz/newdisk2/for_lpy/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/ymz/newdisk2/for_lpy/data/lasot_lmdb'
    settings.lasot_path = '/home/ymz/newdisk2/for_lpy/data/lasot'
    settings.network_path = '/home/ymz/newdisk2/for_lpy/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/ymz/newdisk2/for_lpy/data/nfs'
    settings.otb_path = '/home/ymz/newdisk2/for_lpy/data/otb'
    settings.prj_dir = '/home/ymz/newdisk2/for_lpy'
    settings.result_plot_path = '/home/ymz/newdisk2/for_lpy/output/test/result_plots'
    settings.results_path = '/home/ymz/newdisk2/for_lpy/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/ymz/newdisk2/for_lpy/output'
    settings.segmentation_path = '/home/ymz/newdisk2/for_lpy/output/test/segmentation_results'
    settings.tc128_path = '/home/ymz/newdisk2/for_lpy/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/ymz/newdisk2/for_lpy/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/ymz/newdisk2/for_lpy/data/trackingnet'
    settings.uav_path = '/home/ymz/newdisk2/for_lpy/data/uav'
    settings.vot18_path = '/home/ymz/newdisk2/for_lpy/data/vot2018'
    settings.vot22_path = '/home/ymz/newdisk2/for_lpy/data/vot2022'
    settings.vot_path = '/home/ymz/newdisk2/for_lpy/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

