"""
Utility functions for patch generators.
"""
import yaml

from ..generator.batch.batchsource import BatchSource
from ..adapters.batchadapterbuilder import BatchAdapterBuilder
# from ..generator.batch import xmlbatchgenerator as dptxmlbatchgenerator
from ..generator.batch import batchgenerator as dptbatchgenerator


def get_config_from_yaml(config_path):
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.load(stream=param_file)

    return parameters['model'], parameters['system'], parameters['training'], parameters['data'], \
           parameters['augmentation']


def _constructgenerator(data_source, adapter, data_config, system_config, training_config, generator_key):
    buffer_size = training_config['iterations'][generator_key]['iteration count'] * \
                  training_config['iterations'][generator_key]['batch size']

    generator = dptbatchgenerator.BatchGenerator(label_dist=data_config['labels'][generator_key]['label ratios'],
                                                 patch_shapes=data_config['images']['patch shapes'],
                                                 spacing_tolerance=data_config['spacing tolerance'],
                                                 mask_spacing=data_config['labels'][generator_key][
                                                     'mask pixel spacing'],
                                                 input_channels=data_config['images']['channels'],
                                                 dimension_order='BHWC',
                                                 label_mode=data_config['labels']['label mode'],
                                                 patch_sources=data_source.collection(purpose_id=generator_key,
                                                                                      category_id=None,
                                                                                      replace=True),
                                                 data_adapter=adapter,
                                                 category_dist=data_config['categories'],
                                                 strict_selection=data_config['labels'][generator_key]['strict selection'],
                                                 create_stats=False,
                                                 main_buffer_size=buffer_size,
                                                 buffer_chunk_size=training_config['buffer chunk size'],
                                                 read_buffer_size=buffer_size,
                                                 multi_threaded=system_config['multi-threaded'],
                                                 process_count=system_config[generator_key]['process count'],
                                                 sampler_count=system_config[generator_key]['sampler count'],
                                                 join_timeout=system_config['join timeout secs'],
                                                 response_timeout=system_config['response timeout secs'],
                                                 poll_timeout=system_config['poll timeout secs'],
                                                 name_tag=generator_key)

    return generator


def _constructdataadapter(data_config, augmentation_config, generator_key):
    data_adapter_builder = BatchAdapterBuilder()
    data_adapter_builder.setaugmenters(config=augmentation_config.get(generator_key))

    label_config = {key: data_config['labels'][key] for key in data_config['labels'] if
                    key not in ('training', 'validation')}
    label_config.update(data_config['labels'][generator_key])
    data_adapter_builder.setlabelmapper(config=label_config)

    if data_config['range normalization']['enabled']:
        data_adapter_builder.setrangenormalizer(config=data_config['range normalization'])

    if data_config['weight mapping']['enabled']:
        data_adapter_builder.setweightmapper(config=data_config['weight mapping'])

    adapter = data_adapter_builder.build()
    return adapter


def get_generator_from_config(config_path, data_config_path, generator_key=None, type='xml'):
    _, system_config, training_config, data_config, augmentation_config = get_config_from_yaml(config_path)

    batch_source = BatchSource(source_items=None)
    batch_source.load(file_path=data_config_path)
    if batch_source.purposes:
        if not batch_source.validate(purpose_distribution=data_config['purposes']):
            raise ValueError("invalid purpose distribution")
    else:
        batch_source.distribute(purpose_distribution=data_config['purposes'])
    #
    adapter = _constructdataadapter(data_config, augmentation_config, generator_key)

    generator = _constructgenerator(data_source=batch_source,
                                    adapter=adapter,
                                    data_config=data_config,
                                    system_config=system_config,
                                    training_config=training_config,
                                    generator_key=generator_key)

    return generator
