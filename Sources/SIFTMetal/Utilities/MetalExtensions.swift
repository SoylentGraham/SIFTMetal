//
//  MetalExtensions.swift
//  SkyLight
//
//  Created by Luke Van In on 2023/01/10.
//

import Foundation
import Metal

func capture(commandQueue: MTLCommandQueue, capture: Bool = true, worker: () -> Void) {
    guard capture else {
        worker()
        return
    }
    let captureManager = MTLCaptureManager.shared()
    let captureDescriptor = MTLCaptureDescriptor()
    captureDescriptor.captureObject = commandQueue
    captureDescriptor.destination = .developerTools
    try! captureManager.startCapture(with: captureDescriptor)
    worker()
    captureManager.stopCapture()
}


func capture(commandQueue: MTLCommandQueue, capture: Bool = true, worker: ()async -> Void) async 
{
	guard capture else {
		await worker()
		return
	}
	let captureManager = MTLCaptureManager.shared()
	let captureDescriptor = MTLCaptureDescriptor()
	captureDescriptor.captureObject = commandQueue
	captureDescriptor.destination = .developerTools
	try! captureManager.startCapture(with: captureDescriptor)
	await worker()
	captureManager.stopCapture()
}
